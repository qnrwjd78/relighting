#!/usr/bin/env python3
"""Train the standalone 60→480 AdapterShadow mask refiner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.shadow_c2f import (  # noqa: E402
    ShadowC2FConfig,
    ShadowC2FLoss,
    ShadowCoarseToFine,
    binary_mask_metrics,
)
from utils.shadow_c2f_dataset import ShadowC2FDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-process batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=0.1)
    parser.add_argument("--coarse-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=0.1)
    parser.add_argument("--coarse-base-channels", type=int, default=32)
    parser.add_argument("--fine-channels", type=int, default=32)
    parser.add_argument("--fine-blocks", type=int, default=4)
    parser.add_argument("--adapter-mode", choices=("required", "optional", "zero"), default="required")
    parser.add_argument(
        "--coarse-prior-mode",
        choices=("adapter_delta", "adapter_target", "physics"),
        default="adapter_delta",
        help="Default uses AdapterShadow target-minus-source coarse probability.",
    )
    parser.add_argument("--detector-dropout", type=float, default=0.1)
    parser.add_argument("--detector-morphology", type=float, default=0.3)
    parser.add_argument("--detector-noise-std", type=float, default=0.03)
    parser.add_argument("--detector-false-blob", type=float, default=0.15)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=260831)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--array-cache-size", type=int, default=16)
    parser.add_argument("--online-frozen-stack", action="store_true")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--wan-weights", type=Path, default=ROOT / "weights/Wan2.2-TI2V-5B")
    parser.add_argument("--baseline-steps", type=int, default=50)
    parser.add_argument("--baseline-cfg-scale", type=float, default=2.0)
    parser.add_argument("--focus-config", type=Path, default=ROOT / "external/FOCUS/configs/sd/sd_dinov2_large.yaml")
    parser.add_argument("--focus-checkpoint", type=Path, default=ROOT / "weights/FOCUS/focus_large_sd.pth")
    return parser.parse_args()


def _distributed(device_spec: str = "cuda") -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        device_type = torch.device(device_spec).type
        backend = "nccl" if device_type == "cuda" else "gloo"
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA training requested but CUDA is unavailable")
        dist.init_process_group(backend=backend)
    return rank, local_rank, world


def _seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _device(args: argparse.Namespace, local_rank: int, world: int) -> torch.device:
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    if world > 1 and requested.type == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return requested


def _amp_context_factory(device: torch.device, precision: str):
    """Return a callable that constructs a fresh autocast/null context."""

    if precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but the selected CUDA device lacks bf16 support")
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext


def _batch_to(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward(model, batch: dict[str, Any]):
    return model(
        batch["image"],
        batch["object_mask"],
        batch["point_map"],
        batch["light_position"],
        batch["coarse_prior"],
        receiver_mask=batch["receiver_mask"],
        adapter_features=batch["adapter_features"],
    )


def _boundary_term(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    dilation = F.max_pool2d(target, 5, stride=1, padding=2)
    erosion = -F.max_pool2d(-target, 5, stride=1, padding=2)
    boundary = (dilation - erosion).clamp(0.0, 1.0) * valid
    denominator = boundary.sum()
    if not bool(denominator > 0):
        return logits.sum() * 0.0
    values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (values * boundary).sum() / denominator


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _reduce(values: torch.Tensor, world: int) -> torch.Tensor:
    if world > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


class DistributedEvalSampler(Sampler[int]):
    """Shard validation indices without DistributedSampler's padding duplicates."""

    def __init__(self, dataset, rank: int, world: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world = int(world)
        if not 0 <= self.rank < self.world:
            raise ValueError(f"invalid distributed rank/world: {self.rank}/{self.world}")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world - 1) // self.world)


@torch.inference_mode()
def validate(
    model,
    loader: DataLoader,
    criterion: ShadowC2FLoss,
    device: torch.device,
    amp_context,
    world: int,
    online_conditioner=None,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(9, dtype=torch.float64, device=device)
    for batch in loader:
        batch = _batch_to(batch, device)
        if online_conditioner is not None:
            batch["adapter_features"], batch["coarse_prior"] = online_conditioner(batch)
        valid = batch["receiver_mask"] * (1.0 - batch["object_mask"])
        with amp_context():
            outputs = _forward(model, batch)
            loss, _ = criterion(outputs, batch["target"], valid)
        metrics = binary_mask_metrics(
            outputs["logits"].float(),
            batch["target"],
            valid_mask=valid,
            reduction="none",
        )
        count = int(batch["target"].shape[0])
        totals[0] += float(loss) * count
        for index, name in enumerate(("dice", "iou", "precision", "recall", "specificity", "accuracy", "ber"), 1):
            totals[index] += metrics[name].double().sum()
        totals[8] += count
    totals = _reduce(totals, world)
    count = totals[8].clamp_min(1.0)
    names = ("loss", "dice", "iou", "precision", "recall", "specificity", "accuracy", "ber")
    return {name: float((totals[index] / count).cpu()) for index, name in enumerate(names)}


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive")
    if args.online_frozen_stack and args.batch_size != 1:
        raise ValueError("online-frozen-stack requires per-process --batch-size 1")
    if args.online_frozen_stack and not args.baseline_checkpoint:
        raise ValueError("online-frozen-stack requires --baseline-checkpoint")
    if args.num_workers < 0 or args.save_every <= 0 or args.log_every <= 0:
        raise ValueError("num-workers must be non-negative; save-every/log-every must be positive")
    rank, local_rank, world = _distributed(args.device)
    _seed(args.seed, rank)
    device = _device(args, local_rank, world)
    main_process = rank == 0
    output_dir = args.output_dir.resolve()
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    augment_options = {
        "dropout_probability": args.detector_dropout,
        "morphology_probability": args.detector_morphology,
        "noise_std": args.detector_noise_std,
        "false_blob_probability": args.detector_false_blob,
    }
    train_dataset = ShadowC2FDataset(
        args.train_manifest.resolve(),
        adapter_mode="zero" if args.online_frozen_stack else args.adapter_mode,
        coarse_prior_mode=args.coarse_prior_mode,
        augment_adapter=not args.online_frozen_stack,
        array_cache_size=args.array_cache_size,
        augment_options=augment_options,
    )
    val_dataset = ShadowC2FDataset(
        args.val_manifest.resolve(),
        adapter_mode="zero" if args.online_frozen_stack else args.adapter_mode,
        coarse_prior_mode=args.coarse_prior_mode,
        augment_adapter=False,
        array_cache_size=args.array_cache_size,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if world > 1 else None
    val_sampler = DistributedEvalSampler(val_dataset, rank, world) if world > 1 else None
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_options,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training loader has zero batches; reduce --batch-size or provide more samples"
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_options,
    )

    config = ShadowC2FConfig(
        coarse_size=(60, 60),
        coarse_base_channels=args.coarse_base_channels,
        fine_channels=args.fine_channels,
        fine_blocks=args.fine_blocks,
        use_receiver_mask=True,
        adapter_feature_channels=1,
    )
    raw_model = ShadowCoarseToFine(config).to(device)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = ShadowC2FLoss(
        dice_weight=args.dice_weight,
        coarse_weight=args.coarse_weight,
        positive_weight=args.positive_weight,
    )
    start_epoch, best_dice, global_step = 0, -1.0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != "tokenlight_shadow_c2f_checkpoint_v1":
            raise ValueError(f"Unexpected checkpoint schema: {args.resume}")
        if checkpoint.get("coarse_prior_mode") != args.coarse_prior_mode:
            raise ValueError(
                "Resume checkpoint coarse_prior_mode does not match requested mode: "
                f"{checkpoint.get('coarse_prior_mode')!r} != {args.coarse_prior_mode!r}"
            )
        resume_config = ShadowC2FConfig(**{
            **checkpoint["model_config"],
            "coarse_size": tuple(checkpoint["model_config"]["coarse_size"]),
        })
        if resume_config != config:
            raise ValueError(
                f"Resume checkpoint model_config does not match requested config: "
                f"{resume_config.to_dict()} != {config.to_dict()}"
            )
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        best_dice = float(checkpoint.get("best_val_dice", -1.0))
        global_step = int(checkpoint.get("global_step", 0))
    if world > 1:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    else:
        model = raw_model
    online_conditioner = None
    if args.online_frozen_stack:
        from utils.shadow_c2f_online import FrozenOnlineShadowConditioner
        online_conditioner = FrozenOnlineShadowConditioner(
            device=device,
            baseline_checkpoint=args.baseline_checkpoint.resolve(),
            wan_weights=args.wan_weights.resolve(),
            focus_config=args.focus_config.resolve(),
            focus_checkpoint=args.focus_checkpoint.resolve(),
            steps=args.baseline_steps,
            cfg_scale=args.baseline_cfg_scale,
        )
    amp_context = _amp_context_factory(device, args.precision)
    writer = None
    if main_process:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(output_dir / "tensorboard")
        except Exception:
            writer = None
        resolved = vars(args).copy()
        resolved.update(
            {
                "world_size": world,
                "global_batch_size": args.batch_size * world,
                "effective_global_batch_size": args.batch_size * world * args.gradient_accumulation_steps,
                "model_config": config.to_dict(),
                "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
            }
        )
        (output_dir / "config_resolved.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.time()
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch = _batch_to(batch, device)
            if online_conditioner is not None:
                batch["adapter_features"], batch["coarse_prior"] = online_conditioner(batch)
            valid = batch["receiver_mask"] * (1.0 - batch["object_mask"])
            with amp_context():
                outputs = _forward(model, batch)
                loss, components = criterion(outputs, batch["target"], valid)
                boundary = _boundary_term(outputs["logits"], batch["target"], valid)
                total = loss + args.boundary_weight * boundary
            (total / args.gradient_accumulation_steps).backward()
            update = step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            if update:
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            count = int(batch["target"].shape[0])
            running += float(total.detach()) * count
            seen += count
            if main_process and writer is not None:
                writer.add_scalar("train/loss", float(total.detach()), global_step)
                writer.add_scalar("train/boundary", float(boundary.detach()), global_step)
                for name, value in components.items():
                    writer.add_scalar(f"train/{name}", float(value.detach()), global_step)
            if main_process and step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={running / max(1, seen):.5f}",
                    flush=True,
                )
        scheduler.step()
        metrics = validate(model, val_loader, criterion, device, amp_context, world, online_conditioner)
        train_totals = torch.tensor([running, seen], dtype=torch.float64, device=device)
        train_totals = _reduce(train_totals, world)
        train_loss = float((train_totals[0] / train_totals[1].clamp_min(1.0)).cpu())
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - epoch_start,
            **{f"val_{name}": value for name, value in metrics.items()},
        }
        if main_process:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if writer is not None:
                for name, value in record.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"epoch/{name}", value, epoch + 1)
            state = {
                "schema": "tokenlight_shadow_c2f_checkpoint_v1",
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_dice": max(best_dice, metrics["dice"]),
                "threshold": 0.5,
                "coarse_prior_mode": args.coarse_prior_mode,
                "model_config": config.to_dict(),
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "training_args": vars(args),
                "validation": metrics,
            }
            _atomic_checkpoint(output_dir / "latest.pt", state)
            if (epoch + 1) % args.save_every == 0:
                _atomic_checkpoint(output_dir / f"epoch-{epoch + 1:03d}.pt", state)
            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                _atomic_checkpoint(output_dir / "best.pt", state)
        if world > 1:
            dist.barrier()

    if writer is not None:
        writer.close()
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
