#!/usr/bin/env python3
"""Prepare a dense LiveLight depth map from a MoGe camera-space point map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-map", type=Path, required=True)
    parser.add_argument(
        "--pbr-depth-png",
        type=Path,
        default=None,
        help="Optional rendered 8-bit depth visualization to audit; it is never used as metric depth.",
    )
    parser.add_argument("--output-depth", type=Path, required=True)
    parser.add_argument("--output-vis", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--anchor-u", type=float, default=0.5)
    parser.add_argument("--anchor-v", type=float, default=0.5)
    parser.add_argument("--anchor-radius", type=int, default=3)
    parser.add_argument("--canonical-anchor-depth", type=float, default=256.0)
    return parser.parse_args()


def finite_percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p01", "p05", "p50", "p95", "p99", "max"),
            np.percentile(values, (0, 1, 5, 50, 95, 99, 100)),
        )
    }


def read_anchor(depth: np.ndarray, valid: np.ndarray, u: float, v: float, radius: int) -> float:
    height, width = depth.shape
    x = int(round(float(u) * max(width - 1, 1)))
    y = int(round(float(v) * max(height - 1, 1)))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = depth[y0:y1, x0:x1]
    patch_valid = valid[y0:y1, x0:x1]
    values = patch[patch_valid]
    if values.size == 0:
        raise ValueError("The requested anchor patch has no valid MoGe points")
    return float(np.median(values))


def colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = depth[valid]
    near, far = np.percentile(values, (1, 99))
    normalized = np.clip((depth - near) / max(float(far - near), 1e-8), 0.0, 1.0)
    # Near geometry is warm and far geometry is cool; invalid pixels stay black.
    red = np.clip(1.5 - 2.0 * normalized, 0.0, 1.0)
    green = np.clip(1.5 - 2.0 * np.abs(normalized - 0.5), 0.0, 1.0)
    blue = np.clip(2.0 * normalized - 0.5, 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb[~valid] = 0.0
    return np.round(rgb * 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    points = np.load(args.point_map).astype(np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 point map, got {points.shape}")

    raw_depth = points[..., 2]
    raw_valid = np.isfinite(points).all(axis=-1) & np.isfinite(raw_depth) & (raw_depth > 0.0)
    if not raw_valid.any():
        raise ValueError("Point map contains no positive finite camera-space z values")

    raw_anchor = read_anchor(
        raw_depth,
        raw_valid,
        args.anchor_u,
        args.anchor_v,
        args.anchor_radius,
    )
    scale = float(args.canonical_anchor_depth) / raw_anchor
    # LiveLight expects a dense positive relative-depth map. The MoGe point map marks
    # invalid rays with zero, so use a robust far depth for those pixels.
    far_fill_raw = float(np.percentile(raw_depth[raw_valid], 99.0))
    dense_raw = np.where(raw_valid, raw_depth, far_fill_raw).astype(np.float32)

    dense_scaled = dense_raw * scale
    depth_image = Image.fromarray(dense_scaled).resize(
        (args.width, args.height), Image.Resampling.BILINEAR
    )
    depth = np.asarray(depth_image, dtype=np.float32)
    valid_image = Image.fromarray(raw_valid.astype(np.uint8) * 255).resize(
        (args.width, args.height), Image.Resampling.NEAREST
    )
    valid = np.asarray(valid_image, dtype=np.uint8) > 0

    if depth.shape != (args.height, args.width):
        raise RuntimeError(f"Unexpected resized depth shape: {depth.shape}")
    if not np.isfinite(depth).all() or float(depth.min()) <= 0.0:
        raise RuntimeError("Prepared depth must be finite and strictly positive")

    args.output_depth.parent.mkdir(parents=True, exist_ok=True)
    args.output_vis.parent.mkdir(parents=True, exist_ok=True)
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_depth, depth)
    Image.fromarray(colorize_depth(depth, valid)).save(args.output_vis)
    Image.fromarray(valid.astype(np.uint8) * 255).save(args.output_mask)

    prepared_anchor = read_anchor(
        depth,
        np.ones_like(valid, dtype=bool),
        args.anchor_u,
        args.anchor_v,
        args.anchor_radius,
    )
    report = {
        "point_map": str(args.point_map.resolve()),
        "point_map_convention": "MoGe camera-space XYZ; depth is positive camera-space Z",
        "input_shape": list(points.shape),
        "output_shape": list(depth.shape),
        "output_dtype": str(depth.dtype),
        "valid_fraction": float(raw_valid.mean()),
        "raw_valid_depth": finite_percentiles(raw_depth[raw_valid]),
        "raw_anchor_depth": raw_anchor,
        "canonical_anchor_depth": float(args.canonical_anchor_depth),
        "scale": scale,
        "invalid_fill_rule": "99th percentile of valid raw camera-space Z before scaling",
        "invalid_fill_raw": far_fill_raw,
        "prepared_depth": finite_percentiles(depth.ravel()),
        "prepared_anchor_depth": prepared_anchor,
    }
    if args.pbr_depth_png is not None:
        pbr_image = Image.open(args.pbr_depth_png)
        pbr_array = np.asarray(pbr_image)
        report["pbr_depth_visualization_audit"] = {
            "path": str(args.pbr_depth_png.resolve()),
            "mode": pbr_image.mode,
            "shape": list(pbr_array.shape),
            "dtype": str(pbr_array.dtype),
            "min": int(pbr_array.min()),
            "max": int(pbr_array.max()),
            "used_for_metric_depth": False,
            "reason": "8-bit rendered visualization; MoGe camera-space Z preserves float metric geometry",
        }
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
