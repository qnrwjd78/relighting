#!/usr/bin/env python3
"""Train physical-task baselines on TokenLight single-light PNG samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.physical_tasks.data import (
    LightPairDataset,
    lightnet_input,
)
from model.physical_tasks.lightnet import LightNet
from model.physical_tasks.models import count_parameters


ROOT = Path(__file__).resolve().parents[2]
def parse_args(argv: list[str] | None = None, *, forced_model: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    if forced_model is None:
        parser.add_argument("--model", required=True, choices=("lightnet",))
    else:
        parser.set_defaults(model=forced_model)
    parser.add_argument("--manifest", default="data_train/physical_tasks_single_light/metadata.jsonl")
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def lightnet_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    delta = batch["light_position"] - batch["camera_location"]
    gt_distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-6)
    gt_direction = delta / gt_distance[:, None]
    direction_loss = (1.0 - (output["direction"] * gt_direction).sum(dim=1)).mean()
    distance_loss = torch.nn.functional.smooth_l1_loss(output["log_distance"], gt_distance.log())
    intensity_loss = torch.nn.functional.smooth_l1_loss(
        output["log_intensity"], batch["light_intensity"].clamp_min(1e-6).log()
    )
    radius_loss = torch.nn.functional.smooth_l1_loss(
        output["log_radius"], batch["light_radius"].clamp_min(1e-6).log()
    )
    color_loss = torch.nn.functional.l1_loss(output["color"], batch["light_color"])
    ambient_loss = torch.nn.functional.smooth_l1_loss(output["ambient_scale"], batch["ambient_scale"])
    safe_log_distance = output["log_distance"].clamp(-4.0, 4.0)
    predicted_position = batch["camera_location"] + output["direction"] * safe_log_distance.exp()[:, None]
    position_loss = torch.nn.functional.smooth_l1_loss(predicted_position, batch["light_position"])
    loss = (
        direction_loss
        + distance_loss
        + 0.5 * position_loss
        + 0.5 * intensity_loss
        + 0.25 * radius_loss
        + 0.5 * color_loss
        + 0.25 * ambient_loss
    )
    angle = torch.rad2deg(torch.acos((output["direction"] * gt_direction).sum(dim=1).clamp(-1.0, 1.0))).mean()
    metrics = {
        "loss": float(loss.detach()),
        "direction_cos": float(direction_loss.detach()),
        "angle_deg": float(angle.detach()),
        "position_l1": float(position_loss.detach()),
        "intensity_log_l1": float(intensity_loss.detach()),
        "color_l1": float(color_loss.detach()),
    }
    return loss, metrics


def create_model(args: argparse.Namespace) -> tuple[torch.nn.Module, Callable, Callable]:
    return LightNet(6, args.base_channels), lightnet_input, lightnet_loss


def average_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    input_builder: Callable,
    loss_function: Callable,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(input_builder(batch))
            _, metrics = loss_function(output, batch)
        batch_size = batch["target"].shape[0]
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        count += batch_size
    return average_metrics(totals, count)


def main(argv: list[str] | None = None, *, forced_model: str | None = None) -> int:
    args = parse_args(argv, forced_model=forced_model)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp = device.type == "cuda" and not args.no_amp
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dataset_class = LightPairDataset
    train_dataset = dataset_class(args.manifest, image_size=args.image_size, split=args.split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=len(train_dataset) >= args.batch_size,
    )
    val_loader = None
    if args.split == "train":
        try:
            val_dataset = dataset_class(args.manifest, image_size=args.image_size, split="val")
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
        except ValueError:
            val_loader = None

    model, input_builder, loss_function = create_model(args)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_loss = 0, math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    print(
        f"model={args.model} parameters={count_parameters(model):,} train_rows={len(train_dataset)} "
        f"device={device} amp={amp}",
        flush=True,
    )
    log_path = output_dir / "metrics.csv"
    write_header = not log_path.exists() or start_epoch == 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals: dict[str, float] = {}
        count = 0
        started = time.time()
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(input_builder(batch))
                loss, metrics = loss_function(output, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = batch["target"].shape[0]
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            count += batch_size
            if step % args.log_every == 0:
                print(f"epoch={epoch:03d} step={step:05d}/{len(train_loader):05d} loss={metrics['loss']:.5f}", flush=True)
        scheduler.step()
        train_metrics = average_metrics(totals, count)
        val_metrics = validate(model, val_loader, input_builder, loss_function, device, amp) if val_loader else {}
        monitored = val_metrics.get("loss", train_metrics["loss"])
        is_best = monitored < best_loss
        best_loss = min(best_loss, monitored)
        state = {
            "epoch": epoch,
            "model_name": args.model,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "args": vars(args),
        }
        torch.save(state, output_dir / "last.pt")
        if is_best:
            torch.save(state, output_dir / "best.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, output_dir / f"epoch-{epoch:03d}.pt")
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
