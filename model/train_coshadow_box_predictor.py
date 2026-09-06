#!/usr/bin/env python3
"""Train the fixed32 light-conditioned CoShadow shadow-box predictor."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import accelerate
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.coshadow_box_predictor import (  # noqa: E402
    CoShadowBoxPredictor,
    CoShadowBoxPredictorConfig,
    coshadow_box_loss,
)


DEFAULT_CONFIG = "configs/train_480/coshadow_box.json"
_FIXED32_LIGHT_FIELDS = ("x", "y", "z", "r", "g", "b", "lambda", "d")


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must be a JSON object: {path}")
    return loaded


def _flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def visit(value: Mapping[str, Any]) -> None:
        for key, item in value.items():
            if key == "launch":
                continue
            if isinstance(item, Mapping):
                visit(item)
            else:
                flat[str(key)] = item

    visit(config)
    return flat


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--train_mode", choices=("single", "zero3"), default="single")
    parser.add_argument("--train_metadata", default=None)
    parser.add_argument("--val_metadata", default=None)
    parser.add_argument("--dataset_base_path", default="data/objaverse_fixed32_png")
    parser.add_argument("--source_key", default="input_image")
    parser.add_argument("--object_mask_key", default="mask")
    parser.add_argument("--attrs_key", default="attrs_json")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dataset_num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=("no", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=260302743)
    parser.add_argument("--output_path", default="outputs/train/coshadow_box_fixed32")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--validate_dataset_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--validate_dataset_samples",
        type=int,
        default=128,
        help="Number of samples per split to fully decode in validation-only mode; 0 means all.",
    )
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--iou_weight", type=float, default=1.0)
    parser.add_argument("--presence_weight", type=float, default=1.0)
    parser.add_argument("--visual_channels", default="32,64,128,256")
    parser.add_argument("--light_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--fourier_features", type=int, default=64)
    parser.add_argument("--fourier_sigma", type=float, default=5.0)
    parser.add_argument("--max_lights", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_coordconv", action=argparse.BooleanOptionalAction, default=True)
    return parser


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=os.environ.get("COSHADOW_BOX_CONFIG", DEFAULT_CONFIG))
    pre_args, _ = pre.parse_known_args()
    raw = _read_json(pre_args.config)
    parser = _parser()
    flat = _flatten_config(raw)
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(flat) - known)
    if unknown:
        parser.error(f"Unknown config key(s): {', '.join(unknown)}")
    parser.set_defaults(**flat, config=pre_args.config)
    args = parser.parse_args()
    if not args.train_metadata or not args.val_metadata:
        parser.error("train_metadata and val_metadata are required via config or CLI")
    if args.train_mode not in {"single", "zero3"}:
        parser.error(f"invalid train_mode: {args.train_mode}")
    if args.mixed_precision not in {"no", "bf16"}:
        parser.error(f"invalid mixed_precision: {args.mixed_precision}")
    for name in (
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "l1_weight",
        "iou_weight",
        "presence_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and non-negative")
    for name in ("image_size", "batch_size", "gradient_accumulation_steps", "num_epochs"):
        if int(getattr(args, name)) < 1:
            parser.error(f"{name} must be positive")
    for name in ("light_dim", "hidden_dim", "fourier_features", "max_lights"):
        if int(getattr(args, name)) < 1:
            parser.error(f"{name} must be positive")
    if int(args.dataset_num_workers) < 0 or int(args.save_steps) < 0:
        parser.error("dataset_num_workers and save_steps must be non-negative")
    if not math.isfinite(float(args.dropout)) or not 0 <= float(args.dropout) < 1:
        parser.error("dropout must be finite and in [0,1)")
    if not math.isfinite(float(args.fourier_sigma)) or float(args.fourier_sigma) <= 0:
        parser.error("fourier_sigma must be finite and positive")
    channels = [item.strip() for item in str(args.visual_channels).split(",") if item.strip()]
    try:
        parsed_channels = [int(item) for item in channels]
    except ValueError:
        parser.error("visual_channels must be a comma-separated list of integers")
    if not parsed_channels or any(value < 1 for value in parsed_channels):
        parser.error("visual_channels must contain positive integers")
    if args.l1_weight + args.iou_weight <= 0:
        parser.error("At least one box regression loss weight must be positive")
    if int(args.validate_dataset_samples) < 0:
        parser.error("validate_dataset_samples must be non-negative")
    return args, raw


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _resolve_image(root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing image key {key!r}")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    root = root.resolve(strict=True)
    path = path.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image path for {key!r} escapes dataset root: {value!r}") from exc
    if path.suffix.lower() != ".png":
        raise ValueError(f"Expected PNG for {key!r}, got {value!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_attrs(value: Any, *, index: int) -> None:
    try:
        attrs = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"Row {index} has invalid attrs JSON") from exc
    if not isinstance(attrs, Mapping):
        raise ValueError(f"Row {index} attrs must be an object")
    lights = attrs.get("lights")
    if not isinstance(lights, list) or not lights or not isinstance(lights[0], Mapping):
        raise ValueError(f"Row {index} attrs must contain lights[0]")
    for name in _FIXED32_LIGHT_FIELDS:
        try:
            number = float(lights[0].get(name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {index} lights[0].{name} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"Row {index} lights[0].{name} must be finite")


def _image_tensor(path: Path, *, size: int, mask: bool) -> torch.Tensor:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L" if mask else "RGB")
        image = image.resize((size, size), resample=Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC)
        channels = 1 if mask else 3
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.reshape(size, size, channels).permute(2, 0, 1).float().div_(255.0)
    return tensor if mask else tensor.mul_(2.0).sub_(1.0)


class CoShadowBoxDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], args: argparse.Namespace, expected_split: str) -> None:
        self.rows = rows
        self.root = Path(args.dataset_base_path)
        if not self.root.is_absolute():
            self.root = REPO_ROOT / self.root
        self.source_key = str(args.source_key)
        self.object_mask_key = str(args.object_mask_key)
        self.attrs_key = str(args.attrs_key)
        self.image_size = int(args.image_size)
        for index, row in enumerate(rows):
            split = row.get("coshadow_split")
            if split != expected_split:
                raise ValueError(f"Row {index} has coshadow_split={split!r}, expected {expected_split!r}")
            bbox = row.get("coshadow_bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Row {index} is missing coshadow_bbox_xyxy")
            try:
                bbox = [float(value) for value in bbox]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {index} bbox must be numeric") from exc
            if any(not math.isfinite(value) or value < 0 or value > 1 for value in bbox):
                raise ValueError(f"Row {index} bbox must be finite and normalized")
            valid = row.get("coshadow_bbox_valid")
            if not isinstance(valid, bool):
                raise ValueError(f"Row {index} coshadow_bbox_valid must be boolean")
            if valid and not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                raise ValueError(f"Row {index} valid bbox must have positive area")
            if not valid and bbox != [0.0, 0.0, 0.0, 0.0]:
                raise ValueError(f"Row {index} empty bbox must be all zeros")
            if self.attrs_key not in row:
                raise ValueError(f"Row {index} is missing {self.attrs_key!r}")
            _validate_attrs(row[self.attrs_key], index=index)
            scene = row.get("scene_id", row.get("scene_folder"))
            if not isinstance(scene, str) or not scene:
                raise ValueError(f"Row {index} is missing scene_id/scene_folder")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "source": _image_tensor(
                _resolve_image(self.root, row.get(self.source_key), self.source_key),
                size=self.image_size,
                mask=False,
            ),
            "object_mask": _image_tensor(
                _resolve_image(self.root, row.get(self.object_mask_key), self.object_mask_key),
                size=self.image_size,
                mask=True,
            ),
            "bbox": torch.tensor(row["coshadow_bbox_xyxy"], dtype=torch.float32),
            "valid": torch.tensor(bool(row.get("coshadow_bbox_valid", False))),
            "attrs": row[self.attrs_key],
            "scene_id": str(row.get("scene_id", row.get("scene_folder", ""))),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source": torch.stack([item["source"] for item in batch]),
        "object_mask": torch.stack([item["object_mask"] for item in batch]),
        "bbox": torch.stack([item["bbox"] for item in batch]),
        "valid": torch.stack([item["valid"] for item in batch]),
        "attrs": [item["attrs"] for item in batch],
        "scene_id": [item["scene_id"] for item in batch],
    }


def _assert_scene_disjoint(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    def scenes(rows: list[dict[str, Any]]) -> set[str]:
        return {str(row.get("scene_id", row.get("scene_folder", ""))) for row in rows}

    overlap = scenes(train_rows) & scenes(val_rows)
    if overlap:
        raise ValueError(f"Train/val scene leakage ({len(overlap)} scenes), e.g. {sorted(overlap)[:5]}")


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def validate_dataset_samples(dataset: CoShadowBoxDataset, sample_count: int) -> dict[str, Any]:
    count = len(dataset) if int(sample_count) == 0 else min(len(dataset), int(sample_count))
    valid = 0
    scenes: set[str] = set()
    for index in range(count):
        item = dataset[index]
        if item["source"].shape[0] != 3 or item["object_mask"].shape[0] != 1:
            raise ValueError(f"Unexpected sample channels at index {index}")
        if not torch.isfinite(item["bbox"]).all() or bool(((item["bbox"] < 0) | (item["bbox"] > 1)).any()):
            raise ValueError(f"Invalid normalized bbox at index {index}: {item['bbox'].tolist()}")
        parsed = json.loads(item["attrs"]) if isinstance(item["attrs"], str) else item["attrs"]
        if not isinstance(parsed, Mapping) or not parsed.get("lights"):
            raise ValueError(f"Invalid light attrs at index {index}")
        valid += int(item["valid"])
        scenes.add(item["scene_id"])
    return {
        "rows": len(dataset),
        "decoded_samples": count,
        "decoded_valid_boxes": valid,
        "decoded_scenes": len(scenes),
    }


@torch.no_grad()
def evaluate(accelerator, model, dataloader, args) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(8, device=accelerator.device, dtype=torch.float64)
    for batch in dataloader:
        batch = _to_device(batch, accelerator.device)
        with accelerator.autocast():
            outputs = model(batch["source"], batch["object_mask"], batch["attrs"])
            _, metrics = coshadow_box_loss(
                outputs,
                batch["bbox"],
                batch["valid"],
                l1_weight=args.l1_weight,
                iou_weight=args.iou_weight,
                presence_weight=args.presence_weight,
            )
        count = float(batch["source"].shape[0])
        valid_count = batch["valid"].double().sum()
        values = torch.stack([
            metrics["loss"].double() * count,
            metrics["l1"].double() * valid_count,
            metrics["iou_loss"].double() * valid_count,
            metrics["mean_iou"].double() * valid_count,
            metrics["presence_bce"].double() * count,
            metrics["presence_accuracy"].double() * count,
            torch.tensor(count, device=accelerator.device, dtype=torch.float64),
            valid_count,
        ])
        totals += values
    totals = accelerator.reduce(totals, reduction="sum")
    count = totals[6].clamp_min(1.0)
    valid_count = totals[7].clamp_min(1.0)
    l1 = totals[1] / valid_count
    iou_loss = totals[2] / valid_count
    mean_iou = totals[3] / valid_count
    presence_bce = totals[4] / count
    loss = args.l1_weight * l1 + args.iou_weight * iou_loss + args.presence_weight * presence_bce
    model.train()
    return {
        "loss": float(loss.item()),
        "l1": float(l1.item()),
        "iou_loss": float(iou_loss.item()),
        "mean_iou": float(mean_iou.item()),
        "presence_bce": float(presence_bce.item()),
        "presence_accuracy": float((totals[5] / count).item()),
    }


def _checkpoint_config(args, model: CoShadowBoxPredictor) -> dict[str, Any]:
    return {
        "schema": "coshadow_box_predictor_v1",
        "model": model.config.to_dict(),
        "source_key": args.source_key,
        "object_mask_key": args.object_mask_key,
        "attrs_key": args.attrs_key,
        "image_size": int(args.image_size),
        "adaptation_note": "fixed32 adds Lightoken attrs because one source has many target lights",
    }


def save_checkpoint(accelerator, model, args, label: str) -> None:
    accelerator.wait_for_everyone()
    # DeepSpeed state gathering is collective, so every rank must participate.
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        output = Path(args.output_path) / label
        output.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
        from safetensors.torch import save_file

        save_file(state, str(output / "model.safetensors"))
        unwrapped = accelerator.unwrap_model(model)
        (output / "config.json").write_text(
            json.dumps(_checkpoint_config(args, unwrapped), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    # Keep workers from entering the next collective while rank zero writes.
    accelerator.wait_for_everyone()


def main() -> None:
    args, raw_config = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = _read_jsonl(args.train_metadata)
    val_rows = _read_jsonl(args.val_metadata)
    _assert_scene_disjoint(train_rows, val_rows)
    train_dataset = CoShadowBoxDataset(train_rows, args, "train")
    val_dataset = CoShadowBoxDataset(val_rows, args, "val")
    if args.validate_dataset_only:
        summary = {
            "train": validate_dataset_samples(train_dataset, args.validate_dataset_samples),
            "val": validate_dataset_samples(val_dataset, args.validate_dataset_samples),
            "scene_disjoint": True,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    accelerator_kwargs = {}
    if args.train_mode == "single":
        accelerator_kwargs.update(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    is_deepspeed = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None) is not None
    if args.train_mode == "zero3" and not is_deepspeed:
        raise RuntimeError("train_mode=zero3 requires an Accelerate DeepSpeed plugin/config")
    if args.train_mode == "single" and is_deepspeed:
        raise RuntimeError("train_mode=single cannot run with an active DeepSpeed plugin")
    if is_deepspeed:
        plugin = accelerator.state.deepspeed_plugin
        ds_config = getattr(plugin, "deepspeed_config", None)
        if not isinstance(ds_config, dict):
            raise RuntimeError("DeepSpeed plugin has no mutable deepspeed_config")
        ds_config["train_micro_batch_size_per_gpu"] = int(args.batch_size)
        ds_config["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
        ds_config["train_batch_size"] = (
            int(args.batch_size)
            * int(args.gradient_accumulation_steps)
            * max(1, int(accelerator.num_processes))
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataset_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        drop_last=args.batch_size > 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataset_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    channels = tuple(int(item) for item in str(args.visual_channels).split(",") if item.strip())
    model = CoShadowBoxPredictor(
        CoShadowBoxPredictorConfig(
            visual_channels=channels,
            light_dim=args.light_dim,
            hidden_dim=args.hidden_dim,
            fourier_features=args.fourier_features,
            fourier_sigma=args.fourier_sigma,
            max_lights=args.max_lights,
            dropout=args.dropout,
            use_coordconv=args.use_coordconv,
        )
    )
    if args.resume_checkpoint:
        from model.coshadow_box_predictor import _load_tensor_state

        checkpoint = Path(args.resume_checkpoint)
        if checkpoint.is_dir():
            checkpoint = next(
                (
                    candidate
                    for candidate in (checkpoint / "model.safetensors", checkpoint / "model.pt")
                    if candidate.is_file()
                ),
                checkpoint / "model.safetensors",
            )
        state = _load_tensor_state(checkpoint)
        model.load_state_dict(state, strict=True)
    # Keep FP32 AdamW master parameters; Accelerator autocast supplies BF16 forwards.
    model.float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    if accelerator.is_main_process:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        (Path(args.output_path) / "train_config.json").write_text(
            json.dumps({"resolved": vars(args), "source": raw_config}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    optimizer_step = 0
    model.train()
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            batch = _to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(batch["source"], batch["object_mask"], batch["attrs"])
                    loss, metrics = coshadow_box_loss(
                        outputs,
                        batch["bbox"],
                        batch["valid"],
                        l1_weight=args.l1_weight,
                        iou_weight=args.iou_weight,
                        presence_weight=args.presence_weight,
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients and not is_deepspeed and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    optimizer_step += 1
                    if accelerator.is_main_process and optimizer_step % 50 == 0:
                        print(
                            f"step={optimizer_step} loss={float(metrics['loss']):.5f} "
                            f"iou={float(metrics['mean_iou']):.4f} "
                            f"presence_acc={float(metrics['presence_accuracy']):.4f}",
                            flush=True,
                        )
                    if args.save_steps > 0 and optimizer_step % args.save_steps == 0:
                        save_checkpoint(accelerator, model, args, f"step-{optimizer_step:08d}")
        validation = evaluate(accelerator, model, val_loader, args)
        if accelerator.is_main_process:
            print(f"epoch={epoch} validation={json.dumps(validation, sort_keys=True)}", flush=True)
    save_checkpoint(accelerator, model, args, "final")


if __name__ == "__main__":
    main()
