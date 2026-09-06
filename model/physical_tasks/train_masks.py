"""Shared training engine for the dedicated VisibilityNet and ShadowNet entrypoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Fixed32MaskDataset, build_shadow_input, build_visibility_input
from .models import count_parameters
from .shadownet import ShadowNet
from .visibilitynet import VisibilityNet


ROOT = Path(__file__).resolve().parents[2]


def model_input(task: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return build_visibility_input(batch) if task == "visibility" else build_shadow_input(batch)


def parse_args(task: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {task.title()}Net on objaverse_fixed32 geometry-ray masks.")
    parser.add_argument("--manifest", default="data_train/objaverse_fixed32_physical/metadata.jsonl")
    parser.add_argument("--split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=480)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    args.task = task
    return args


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def balanced_bce(logits: torch.Tensor, target: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    logits, target, domain = logits.float(), target.float(), domain.float()
    positive = target * domain
    negative = (1.0 - target) * domain
    positive_loss = (torch.nn.functional.softplus(-logits) * positive).sum() / positive.sum().clamp_min(1.0)
    negative_loss = (torch.nn.functional.softplus(logits) * negative).sum() / negative.sum().clamp_min(1.0)
    return 0.5 * (positive_loss + negative_loss)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    logits, target, domain = logits.float(), target.float(), domain.float()
    probability = torch.sigmoid(logits) * domain
    target = target * domain
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, domain: torch.Tensor) -> dict[str, float]:
    logits, target, domain = logits.float(), target.float(), domain.float()
    prediction = (logits >= 0).float() * domain
    target = target * domain
    intersection = (prediction * target).sum()
    union = ((prediction + target) > 0).float().sum()
    tp = intersection
    precision = tp / prediction.sum().clamp_min(1.0)
    recall = tp / target.sum().clamp_min(1.0)
    iou = intersection / union.clamp_min(1.0)
    return {"iou": float(iou.detach()), "precision": float(precision.detach()), "recall": float(recall.detach())}


def mask_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    if args.task == "visibility":
        logits = output["visibility_logits"]
        target = batch["visibility"]
        domain = batch["mask"]
        bce = balanced_bce(logits, target, domain)
        dice = dice_loss(logits, target, domain)
        loss = bce + args.dice_weight * dice
        return loss, {"loss": float(loss.detach()), "bce": float(bce.detach()), "dice_loss": float(dice.detach()), **segmentation_metrics(logits, target, domain)}

    logits = output["shadow_logits"]
    domain = batch["receiver_mask"]
    bce = balanced_bce(logits, batch["shadow"], domain)
    dice = dice_loss(logits, batch["shadow"], domain)
    loss = bce + args.dice_weight * dice
    return loss, {
        "loss": float(loss.detach()),
        "bce": float(bce.detach()),
        "dice_loss": float(dice.detach()),
        **segmentation_metrics(logits, batch["shadow"], domain),
    }


def mean_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in total.items()}


@torch.no_grad()
def validate(model, loader, args, device, amp) -> dict[str, float]:
    model.eval()
    total: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(model_input(args.task, batch))
            _, metrics = mask_loss(output, batch, args)
        size = batch["source"].shape[0]
        for key, value in metrics.items():
            total[key] = total.get(key, 0.0) + value * size
        count += size
    return mean_metrics(total, count)


def train_main(task: str) -> int:
    if task not in {"visibility", "shadow"}:
        raise ValueError(task)
    args = parse_args(task)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp = device.type == "cuda" and not args.no_amp
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    dataset = Fixed32MaskDataset(args.manifest, image_size=args.image_size, split=args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        drop_last=len(dataset) >= args.batch_size,
    )
    val_loader = None
    if args.split == "train":
        val_dataset = Fixed32MaskDataset(args.manifest, image_size=args.image_size, split="val")
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
        )
    in_channels = 15 if task == "visibility" else 16
    model = (VisibilityNet if task == "visibility" else ShadowNet)(in_channels, args.base_channels).to(device)
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
    print(f"task={task} parameters={count_parameters(model):,} train_rows={len(dataset)} device={device} amp={amp}", flush=True)

    log_path = output_dir / "metrics.csv"
    write_header = not log_path.exists() or start_epoch == 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total: dict[str, float] = {}
        count = 0
        started = time.time()
        for step, batch in enumerate(loader):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(model_input(task, batch))
                loss, metrics = mask_loss(output, batch, args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            size = batch["source"].shape[0]
            for key, value in metrics.items():
                total[key] = total.get(key, 0.0) + value * size
            count += size
            if step % args.log_every == 0:
                print(f"epoch={epoch:03d} step={step:05d}/{len(loader):05d} loss={metrics['loss']:.5f} iou={metrics['iou']:.4f}", flush=True)
        scheduler.step()
        train_metrics = mean_metrics(total, count)
        val_metrics = validate(model, val_loader, args, device, amp) if val_loader else {}
        monitored = val_metrics.get("loss", train_metrics["loss"])
        is_best = monitored < best_loss
        best_loss = min(best_loss, monitored)
        state = {
            "epoch": epoch, "task": task, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_loss": best_loss, "args": vars(args),
        }
        torch.save(state, output_dir / "last.pt")
        if is_best:
            torch.save(state, output_dir / "best.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, output_dir / f"epoch-{epoch:03d}.pt")
        record = {
            "epoch": epoch, "seconds": round(time.time() - started, 3),
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
