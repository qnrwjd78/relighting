#!/usr/bin/env python3
"""Infer refined cast-shadow masks and emit a strict Wan-ready manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw
from scipy import ndimage
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.shadow_c2f import ShadowC2FConfig, ShadowCoarseToFine  # noqa: E402
from utils.shadow_c2f_dataset import ShadowC2FDataset  # noqa: E402
from utils.shadow_pipeline_io import (  # noqa: E402
    atomic_write_jsonl,
    sample_key,
    sample_name,
    scene_id,
)


INFERENCE_CACHE_SCHEMA = "tokenlight_shadow_c2f_inference_artifact_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        help="Optional GT-preserving manifest; defaults inside --output-root.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--adapter-mode", choices=("required", "optional", "zero"), default="required")
    parser.add_argument(
        "--coarse-prior-mode",
        choices=("adapter_delta", "adapter_target", "physics"),
        help="Must match the checkpoint; defaults to the checkpoint value.",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--max-hole-area", type=int, default=0)
    parser.add_argument("--fallback", choices=("none", "coarse", "empty"), default="none")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--save-diagnostics", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _model_config(value: dict[str, Any]) -> ShadowC2FConfig:
    fields = dict(value)
    if "coarse_size" in fields:
        fields["coarse_size"] = tuple(int(item) for item in fields["coarse_size"])
    return ShadowC2FConfig(**fields)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_signature(path: str | Path) -> dict[str, Any]:
    value = Path(path).resolve()
    if not value.is_file():
        return {"path": value.as_posix(), "missing": True}
    stat = value.stat()
    return {
        "path": value.as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sample_cache_id(
    row: dict[str, Any],
    run_fingerprint: str,
    adapter_mode: str,
    coarse_prior_mode: str,
) -> str:
    input_keys = [
        "source_image",
        "object_mask",
        "receiver_mask",
        "geometry_cache",
    ]
    if coarse_prior_mode == "physics":
        input_keys.append("physics_cache")
    if adapter_mode != "zero":
        input_keys.extend(
            ("adapter_target_cache", "adapter_source_cache", "adapter_delta_cache")
        )
    inputs = {
        key: _path_signature(str(row.get(key, "")))
        for key in input_keys
    }
    return _fingerprint(
        {
            "schema": INFERENCE_CACHE_SCHEMA,
            "sample_id": sample_key(row),
            "light_position": row.get("light_position"),
            "run_fingerprint": run_fingerprint,
            "inputs": inputs,
        }
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _inference_cache_valid(
    metadata_path: Path,
    cache_id: str,
    probability_path: Path,
    preview_path: Path,
    binary_path: Path,
    expected_shape: tuple[int, int],
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != INFERENCE_CACHE_SCHEMA or metadata.get("cache_id") != cache_id:
            return False
        paths = {
            "probability": probability_path,
            "preview": preview_path,
            "binary": binary_path,
        }
        recorded = metadata.get("artifacts")
        if not isinstance(recorded, dict):
            return False
        if any(recorded.get(name) != _path_signature(path) for name, path in paths.items()):
            return False
        probability = np.load(probability_path, allow_pickle=False)
        if probability.shape != expected_shape or not np.issubdtype(probability.dtype, np.floating):
            return False
        if not np.isfinite(probability).all():
            return False
        if probability.size and (
            float(probability.min()) < -1e-4 or float(probability.max()) > 1.0001
        ):
            return False
        with Image.open(preview_path) as preview:
            if preview.size != (expected_shape[1], expected_shape[0]):
                return False
        with Image.open(binary_path) as binary_image:
            if binary_image.size != (expected_shape[1], expected_shape[0]):
                return False
            binary = np.asarray(binary_image.convert("L"), dtype=np.uint8)
        if not np.isin(binary, (0, 255)).all():
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _postprocess(
    probability: np.ndarray,
    receiver: np.ndarray,
    object_mask: np.ndarray,
    threshold: float,
    min_area: int,
    max_hole_area: int,
) -> np.ndarray:
    binary = (probability >= threshold) & receiver & ~object_mask
    if min_area > 0 and binary.any():
        labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_area
        keep[0] = False
        binary = keep[labels]
    if max_hole_area > 0:
        holes = (~binary) & receiver & ~object_mask
        labels, count = ndimage.label(holes, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel())
        fill = sizes <= max_hole_area
        fill[0] = False
        binary |= fill[labels]
    return binary


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npy", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.save(temporary_name, value)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.png")
    Image.fromarray(value).save(temporary, format="PNG")
    temporary.replace(path)


def _confidence(probability: np.ndarray, receiver: np.ndarray, object_mask: np.ndarray) -> float:
    valid = receiver & ~object_mask
    if not valid.any():
        return 0.0
    margin = np.abs(probability[valid] - 0.5) * 2.0
    return float(np.mean(margin))


def _diagnostic(
    row: dict[str, Any],
    probability: np.ndarray,
    binary: np.ndarray,
    coarse_prior: np.ndarray,
    adapter: np.ndarray,
    path: Path,
) -> None:
    source = Image.open(row["source_image"]).convert("RGB").resize((480, 480))
    fields = [
        source,
        Image.fromarray((adapter * 255).round().astype(np.uint8), mode="L").convert("RGB"),
        Image.fromarray((coarse_prior * 255).round().astype(np.uint8), mode="L").convert("RGB"),
        Image.fromarray((probability * 255).round().astype(np.uint8), mode="L").convert("RGB"),
        Image.fromarray((binary.astype(np.uint8) * 255), mode="L").convert("RGB"),
    ]
    gt_path = Path(str(row.get("gt_shadow_mask", "")))
    if gt_path.is_file():
        fields.append(Image.open(gt_path).convert("RGB").resize((480, 480)))
    labels = ["source", "adapter target", "coarse prior", "refined prob", "refined binary", "GT"]
    panel = Image.new("RGB", (480 * len(fields), 520), "white")
    draw = ImageDraw.Draw(panel)
    for index, image in enumerate(fields):
        panel.paste(ImageOps.fit(image, (480, 480)), (index * 480, 40))
        draw.text((index * 480 + 8, 10), labels[index], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.min_component_area < 0 or args.max_hole_area < 0:
        raise ValueError("Invalid batch-size or morphology area")
    if args.num_workers < 0 or args.limit < 0 or args.save_diagnostics < 0:
        raise ValueError("num-workers, limit, and save-diagnostics must be non-negative")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "tokenlight_shadow_c2f_checkpoint_v1":
        raise ValueError(f"Unexpected checkpoint schema: {args.checkpoint}")
    checkpoint_prior_mode = checkpoint.get("coarse_prior_mode")
    if checkpoint_prior_mode not in {"adapter_delta", "adapter_target", "physics"}:
        raise ValueError(
            "Checkpoint has no valid coarse_prior_mode; retrain it with the "
            "Adapter-C2F contract"
        )
    coarse_prior_mode = args.coarse_prior_mode or str(checkpoint_prior_mode)
    if coarse_prior_mode != checkpoint_prior_mode:
        raise ValueError(
            "--coarse-prior-mode must match the checkpoint: "
            f"{coarse_prior_mode!r} != {checkpoint_prior_mode!r}"
        )
    threshold = float(checkpoint.get("threshold", 0.5) if args.threshold is None else args.threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("min-confidence must be in [0,1]")

    dataset = ShadowC2FDataset(
        args.manifest.resolve(),
        adapter_mode=args.adapter_mode,
        coarse_prior_mode=coarse_prior_mode,
        require_target=False,
        augment_adapter=False,
        array_cache_size=16,
    )
    if args.limit > 0:
        dataset.rows = dataset.rows[: args.limit]
    rows = dataset.rows
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but the selected CUDA device lacks bf16 support")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = ShadowCoarseToFine(_model_config(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    binary_root = output_root / "refined_binary"
    probability_root = output_root / "refined_probability"
    preview_root = output_root / "refined_probability_png"
    diagnostics_root = output_root / "diagnostics"
    metadata_root = output_root / "cache_metadata"
    output_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    row_offset = 0
    fallback_count = 0
    cache_hit_count = 0
    use_amp = args.precision == "bf16" and device.type == "cuda"
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    run_options = {
        "schema": INFERENCE_CACHE_SCHEMA,
        "algorithm": "receiver_nonobject_gated_probability_v1",
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": checkpoint["model_config"],
        "threshold": threshold,
        "min_component_area": int(args.min_component_area),
        "max_hole_area": int(args.max_hole_area),
        "fallback": args.fallback,
        "min_confidence": float(args.min_confidence),
        "adapter_mode": args.adapter_mode,
        "coarse_prior_mode": coarse_prior_mode,
    }
    run_fingerprint = _fingerprint(run_options)

    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with context:
                outputs = model(
                    tensors["image"],
                    tensors["object_mask"],
                    tensors["point_map"],
                    tensors["light_position"],
                    tensors["coarse_prior"],
                    receiver_mask=tensors["receiver_mask"],
                    adapter_features=tensors["adapter_features"],
                )
            probabilities = outputs["mask"].float().cpu().numpy()[:, 0]
            coarse_prior_full = F.interpolate(
                tensors["coarse_prior"].float(),
                size=probabilities.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).cpu().numpy()[:, 0]
            adapter_target = tensors["adapter_features"].float().cpu().numpy()[:, 0]
            objects = tensors["object_mask"].cpu().numpy()[:, 0] >= 0.5
            receivers = tensors["receiver_mask"].cpu().numpy()[:, 0] >= 0.5

            for batch_index in range(probabilities.shape[0]):
                row = dict(rows[row_offset + batch_index])
                expected_sample_id = sample_key(row)
                if str(batch["sample_id"][batch_index]) != expected_sample_id:
                    raise RuntimeError(
                        f"DataLoader/manifest order mismatch: {batch['sample_id'][batch_index]!r} "
                        f"!= {expected_sample_id!r}"
                    )
                scene, sample = scene_id(row), sample_name(row)
                probability_path = probability_root / scene / f"{sample}.npy"
                preview_path = preview_root / scene / f"{sample}.png"
                binary_path = binary_root / scene / f"{sample}.png"
                metadata_path = metadata_root / scene / f"{sample}.json"
                confidence = _confidence(
                    probabilities[batch_index], receivers[batch_index], objects[batch_index]
                )
                support = receivers[batch_index] & ~objects[batch_index]
                probability = np.where(support, probabilities[batch_index], 0.0).clip(0.0, 1.0)
                fallback_used = "none"
                if confidence < args.min_confidence and args.fallback != "none":
                    fallback_used = args.fallback
                    fallback_count += 1
                    probability = (
                        coarse_prior_full[batch_index]
                        if args.fallback == "coarse"
                        else np.zeros_like(probability)
                    )
                    probability = np.where(support, probability, 0.0).clip(0.0, 1.0)
                binary = _postprocess(
                    probability,
                    receivers[batch_index],
                    objects[batch_index],
                    threshold,
                    args.min_component_area,
                    args.max_hole_area,
                )
                cache_id = _sample_cache_id(
                    row,
                    run_fingerprint,
                    args.adapter_mode,
                    coarse_prior_mode,
                )
                cache_valid = args.skip_existing and _inference_cache_valid(
                    metadata_path,
                    cache_id,
                    probability_path,
                    preview_path,
                    binary_path,
                    tuple(int(value) for value in probability.shape),
                )
                if cache_valid:
                    cache_hit_count += 1
                else:
                    _atomic_npy(probability_path, probability.astype(np.float16))
                    _atomic_png(
                        preview_path,
                        (probability.clip(0, 1) * 255.0).round().astype(np.uint8),
                    )
                    _atomic_png(binary_path, binary.astype(np.uint8) * 255)
                    _atomic_json(
                        metadata_path,
                        {
                            "schema": INFERENCE_CACHE_SCHEMA,
                            "cache_id": cache_id,
                            "run_fingerprint": run_fingerprint,
                            "sample_id": expected_sample_id,
                            "artifacts": {
                                "probability": _path_signature(probability_path),
                                "preview": _path_signature(preview_path),
                                "binary": _path_signature(binary_path),
                            },
                        },
                    )
                row["predicted_shadow_mask"] = binary_path.as_posix()
                row["refined_probability"] = probability_path.as_posix()
                row["refined_probability_png"] = preview_path.as_posix()
                row["shadow_mask_cache_metadata"] = metadata_path.as_posix()
                row["shadow_mask_cache_id"] = cache_id
                row["shadow_mask_confidence_heuristic"] = confidence
                row["shadow_mask_fallback"] = fallback_used
                row["shadow_mask_threshold"] = threshold
                # Enforce leakage-safe inference rows even if an older input
                # manifest still contained the renderer field.
                row.pop("shadow_mask", None)
                row.pop("shadow_mask_pad16", None)
                evaluation_row = dict(row)
                evaluation_rows.append(evaluation_row)
                wan_row = dict(row)
                wan_row.pop("gt_shadow_mask", None)
                output_rows.append(wan_row)
                if args.save_diagnostics > 0 and len(output_rows) <= args.save_diagnostics:
                    _diagnostic(
                        evaluation_row,
                        probability,
                        binary,
                        coarse_prior_full[batch_index],
                        adapter_target[batch_index],
                        diagnostics_root / scene / f"{sample}.jpg",
                    )
            row_offset += probabilities.shape[0]
            print(f"refined={row_offset}/{len(rows)}", flush=True)

    output_manifest = (
        args.output_manifest.resolve()
        if args.output_manifest
        else output_root / "wan_manifest.jsonl"
    )
    evaluation_manifest = (
        args.evaluation_manifest.resolve()
        if args.evaluation_manifest
        else output_root / "evaluation_manifest.jsonl"
    )
    if evaluation_manifest == output_manifest:
        raise ValueError("WAN and evaluation manifests must use different paths")
    atomic_write_jsonl(output_manifest, output_rows)
    atomic_write_jsonl(evaluation_manifest, evaluation_rows)
    summary = {
        "schema": "tokenlight_shadow_c2f_inference_v1",
        "input_manifest": args.manifest.resolve().as_posix(),
        "output_manifest": output_manifest.as_posix(),
        "wan_manifest": output_manifest.as_posix(),
        "evaluation_manifest": evaluation_manifest.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "run_fingerprint": run_fingerprint,
        "rows": len(output_rows),
        "evaluation_rows_with_gt": sum(
            bool(row.get("gt_shadow_mask")) for row in evaluation_rows
        ),
        "threshold": threshold,
        "min_component_area": args.min_component_area,
        "max_hole_area": args.max_hole_area,
        "adapter_mode": args.adapter_mode,
        "coarse_prior_mode": coarse_prior_mode,
        "fallback": args.fallback,
        "min_confidence": args.min_confidence,
        "fallback_count": fallback_count,
        "validated_cache_hits": cache_hit_count,
        "gt_conditioning_key_present": False,
    }
    (output_root / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
