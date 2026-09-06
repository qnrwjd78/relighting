#!/usr/bin/env python3
"""Build per-scene point-map caches and optional geometric shadow priors.

The default can reproduce the paper-style geometric prior.  ``--geometry-only``
is the contract used by the TokenLight + AdapterShadow pipeline: AdapterShadow
provides the coarse mask while this script only prepares point-map features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.shadow_c2f import (  # noqa: E402
    PhysicsPriorConfig,
    directional_shadow_prior,
    point_light_shadow_prior,
)
from utils.shadow_geometry import (  # noqa: E402
    canonical_light_to_opencv,
    load_and_align_moge_point_map,
    load_binary_mask,
    normalize_point_features,
    reconstruct_oracle_point_map,
    reference_light_direction,
)
from utils.shadow_c2f_dataset import (  # noqa: E402
    GEOMETRY_CACHE_SCHEMA,
    PHYSICS_CACHE_SCHEMA,
)
from utils.shadow_pipeline_io import (  # noqa: E402
    atomic_write_jsonl,
    light_position,
    read_jsonl,
    sample_name,
    scene_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--geometry-mode", choices=("oracle", "moge", "auto"), default="auto")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Write geometry caches/manifests without a ray-cast physics prior.",
    )
    parser.add_argument("--prior-kind", choices=("point", "directional"), default="point")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--coarse-size", type=int, default=60)
    parser.add_argument("--angular-tolerance", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--blur-sigma", type=float, default=0.8)
    parser.add_argument("--object-chunk-size", type=int, default=128)
    parser.add_argument("--receiver-chunk-size", type=int, default=1024)
    parser.add_argument("--feature-scale", type=float, default=4.0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        Image.fromarray(value, mode="L").save(temporary_name, format="PNG")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_signature(path: str | Path) -> dict[str, Any]:
    value = Path(path).resolve()
    stat = value.stat()
    return {
        "path": value.as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _geometry_cache_id(row: dict[str, Any], mode: str, feature_scale: float) -> str:
    assets = {
        "object_mask": _path_signature(row["object_mask"]),
        "receiver_mask": _path_signature(row["receiver_mask"]),
    }
    if mode == "oracle":
        assets.update(
            {
                "pbr_depth": _path_signature(row["pbr_depth"]),
                "scene_meta": _path_signature(row["scene_meta"]),
            }
        )
    else:
        assets["point_map"] = _path_signature(row["point_map"])
    return _fingerprint(
        {
            "schema": GEOMETRY_CACHE_SCHEMA,
            "algorithm": "tokenlight_geometry_alignment_v1",
            "geometry_mode": mode,
            "feature_scale": float(feature_scale),
            "assets": assets,
        }
    )


def _prior_cache_id(
    geometry_cache_id: str,
    light: list[float],
    prior_kind: str,
    config: PhysicsPriorConfig,
) -> str:
    return _fingerprint(
        {
            "schema": PHYSICS_CACHE_SCHEMA,
            "algorithm": "tokenlight_chunked_angular_prior_v1",
            "geometry_cache_id": geometry_cache_id,
            "light_position_canonical": [float(value) for value in light],
            "prior_kind": prior_kind,
            "physics_config": asdict(config),
        }
    )


def _scalar_text(archive: Any, field: str) -> str:
    value = np.asarray(archive[field])
    if value.shape != ():
        raise ValueError(f"{field} must be scalar")
    return str(value.item())


def _geometry_cache_valid(
    path: Path,
    expected_id: str,
    expected_shape: tuple[int, int],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "schema",
                "cache_id",
                "point_features",
                "point_valid",
                "points_cv_canonical",
                "geometry_mode",
                "feature_scale",
            }
            if not required.issubset(archive.files):
                return False
            if _scalar_text(archive, "schema") != GEOMETRY_CACHE_SCHEMA:
                return False
            if _scalar_text(archive, "cache_id") != expected_id:
                return False
            points = np.asarray(archive["points_cv_canonical"])
            features = np.asarray(archive["point_features"])
            valid = np.asarray(archive["point_valid"])
            expected_points = (3, *expected_shape)
            if points.shape != expected_points or features.shape != expected_points:
                return False
            if valid.shape != (1, *expected_shape):
                return False
            if not np.issubdtype(points.dtype, np.floating) or not np.issubdtype(
                features.dtype, np.floating
            ):
                return False
            if not np.isfinite(points).all() or not np.isfinite(features).all():
                return False
            if not np.isin(valid, (0, 1)).all():
                return False
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return True


def _prior_cache_valid(
    path: Path,
    expected_id: str,
    geometry_cache_id: str,
    coarse_size: tuple[int, int],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "schema",
                "cache_id",
                "geometry_cache_id",
                "coarse_size",
                "physics_prior",
                "light_position_cv",
                "light_direction",
                "geometry_mode",
                "prior_kind",
            }
            if not required.issubset(archive.files):
                return False
            if _scalar_text(archive, "schema") != PHYSICS_CACHE_SCHEMA:
                return False
            if _scalar_text(archive, "cache_id") != expected_id:
                return False
            if _scalar_text(archive, "geometry_cache_id") != geometry_cache_id:
                return False
            if tuple(int(value) for value in np.asarray(archive["coarse_size"])) != coarse_size:
                return False
            prior = np.asarray(archive["physics_prior"])
            direction = np.asarray(archive["light_direction"])
            position = np.asarray(archive["light_position_cv"])
            if prior.shape != coarse_size or direction.shape != (3,) or position.shape != (3,):
                return False
            if not all(np.issubdtype(value.dtype, np.floating) for value in (prior, direction, position)):
                return False
            if not all(np.isfinite(value).all() for value in (prior, direction, position)):
                return False
            if prior.size and (float(prior.min()) < -1e-4 or float(prior.max()) > 1.0001):
                return False
            if float(np.linalg.norm(direction.astype(np.float64))) <= 1e-6:
                return False
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return True


def _write_preview(path: Path, probability: torch.Tensor, output_shape: tuple[int, int]) -> None:
    preview = F.interpolate(
        probability[None, None].float(),
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    _atomic_png(path, (preview.cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8))


def _choose_mode(row: dict[str, Any], requested: str) -> str:
    oracle = Path(str(row.get("scene_meta", ""))).is_file() and Path(
        str(row.get("pbr_depth", ""))
    ).is_file()
    moge = Path(str(row.get("point_map", ""))).is_file()
    if requested == "oracle":
        if not oracle:
            raise FileNotFoundError(f"Oracle geometry unavailable for {scene_id(row)}")
        return "oracle"
    if requested == "moge":
        if not moge:
            raise FileNotFoundError(f"MoGe geometry unavailable for {scene_id(row)}")
        return "moge"
    if oracle:
        return "oracle"
    if moge:
        return "moge"
    raise FileNotFoundError(f"No geometry available for {scene_id(row)}")


def _load_scene_geometry(row: dict[str, Any], mode: str, feature_scale: float):
    object_mask = load_binary_mask(row["object_mask"])
    receiver_mask = load_binary_mask(row["receiver_mask"])
    if object_mask.shape != receiver_mask.shape:
        raise ValueError(f"Object/receiver shape mismatch for {scene_id(row)}")
    height, width = object_mask.shape[-2:]
    if mode == "oracle":
        points, valid, metadata = reconstruct_oracle_point_map(
            row["pbr_depth"], row["scene_meta"], width=width, height=height
        )
    else:
        points, valid, metadata = load_and_align_moge_point_map(
            row["point_map"], object_mask
        )
    # Physics only needs visible object/receiver surfaces.  This also removes
    # finite but unrelated MoGe pixels outside the renderer receiver contract.
    support = ((object_mask + receiver_mask) > 0.5).float()
    valid = valid * support
    points = torch.where(valid.bool().expand_as(points), points, torch.zeros_like(points))
    features = normalize_point_features(points, valid, scale=feature_scale)
    return points, features, valid, object_mask, receiver_mask, metadata


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if not np.isfinite(args.feature_scale) or args.feature_scale <= 0:
        raise ValueError("--feature-scale must be finite and positive")
    rows = read_jsonl(args.manifest.resolve())
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows to process")
    config = PhysicsPriorConfig(
        coarse_size=(args.coarse_size, args.coarse_size),
        angular_tolerance_degrees=args.angular_tolerance,
        temperature=args.temperature,
        object_chunk_size=args.object_chunk_size,
        receiver_chunk_size=args.receiver_chunk_size,
        gaussian_blur_sigma=args.blur_sigma,
    )
    device = torch.device(args.device)
    if not args.geometry_only and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    # Keep only the active scene's dense 480x480 geometry in memory.  Full
    # train manifests can contain thousands of scenes, so retaining every
    # tensor would otherwise consume several gigabytes of host RAM.
    active_scene: str | None = None
    active_state: tuple[Any, ...] | None = None
    scene_cache_ids: dict[str, str] = {}
    generated_geometry = 0
    generated_priors = 0

    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            scene = scene_id(row)
            mode = _choose_mode(row, args.geometry_mode)
            geometry_cache_id = _geometry_cache_id(row, mode, args.feature_scale)
            previous_id = scene_cache_ids.setdefault(scene, geometry_cache_id)
            if previous_id != geometry_cache_id:
                raise ValueError(
                    f"Conflicting geometry assets share scene_id {scene!r}; "
                    "use distinct scene IDs or output roots"
                )
            if active_scene != scene or active_state is None:
                loaded = _load_scene_geometry(row, mode, args.feature_scale)
                points, features, valid, object_mask, receiver_mask, metadata = loaded
                geometry_path = output_root / "geometry" / f"{scene}.npz"
                expected_shape = tuple(int(value) for value in points.shape[-2:])
                geometry_valid = args.skip_existing and _geometry_cache_valid(
                    geometry_path, geometry_cache_id, expected_shape
                )
                if not geometry_valid:
                    _atomic_npz(
                        geometry_path,
                        schema=np.asarray(GEOMETRY_CACHE_SCHEMA),
                        cache_id=np.asarray(geometry_cache_id),
                        point_features=features.numpy().astype(np.float16),
                        point_valid=valid.numpy().astype(np.uint8),
                        points_cv_canonical=points.numpy().astype(np.float16),
                        geometry_mode=np.asarray(mode),
                        feature_scale=np.asarray(args.feature_scale, dtype=np.float32),
                        **{
                            f"meta_{key}": np.asarray(value)
                            for key, value in metadata.items()
                            if isinstance(value, (int, float, str))
                        },
                    )
                    generated_geometry += 1
                active_scene = scene
                active_state = (
                    mode,
                    geometry_path,
                    geometry_cache_id,
                    points,
                    valid,
                    object_mask,
                    receiver_mask,
                    metadata,
                )

            assert active_state is not None
            (
                mode,
                geometry_path,
                geometry_cache_id,
                points,
                valid,
                object_mask,
                receiver_mask,
                metadata,
            ) = active_state
            current = dict(row)
            current["geometry_mode"] = mode
            current["geometry_cache"] = geometry_path.as_posix()
            current["geometry_cache_id"] = geometry_cache_id
            if args.geometry_only:
                # Preserve the exact canonical point-light position on the row.
                # The C2F dataset converts it to normalized OpenCV camera space
                # using the same feature scale as the point map.
                current["light_position"] = light_position(row)
                current.pop("physics_cache", None)
                current.pop("physics_cache_id", None)
                output_rows.append(current)
                if index % 25 == 0 or index == len(rows):
                    print(
                        f"geometry={index}/{len(rows)} "
                        f"generated_scenes={generated_geometry}",
                        flush=True,
                    )
                continue

            prior_path = output_root / "priors" / scene / f"{sample_name(row)}.npz"
            canonical_light = light_position(row)
            physics_cache_id = _prior_cache_id(
                geometry_cache_id,
                canonical_light,
                args.prior_kind,
                config,
            )
            current["physics_cache"] = prior_path.as_posix()
            current["physics_cache_id"] = physics_cache_id
            output_rows.append(current)
            preview_path = prior_path.with_suffix(".png")
            coarse_size = tuple(int(value) for value in config.coarse_size)
            if args.skip_existing and _prior_cache_valid(
                prior_path,
                physics_cache_id,
                geometry_cache_id,
                coarse_size,
            ):
                if not preview_path.is_file():
                    with np.load(prior_path, allow_pickle=False) as archive:
                        cached_probability = torch.from_numpy(
                            np.asarray(archive["physics_prior"], dtype=np.float32).copy()
                        )
                    _write_preview(
                        preview_path,
                        cached_probability,
                        tuple(int(value) for value in object_mask.shape[-2:]),
                    )
                continue

            light_cv = canonical_light_to_opencv(canonical_light)
            direction = reference_light_direction(points, object_mask, valid, light_cv)
            batch_points = points.unsqueeze(0).to(device)
            batch_object = object_mask.unsqueeze(0).to(device)
            batch_receiver = receiver_mask.unsqueeze(0).to(device)
            batch_valid = valid.unsqueeze(0).to(device)
            if args.prior_kind == "point":
                prior = point_light_shadow_prior(
                    batch_points,
                    batch_object,
                    light_cv.to(device),
                    receiver_mask=batch_receiver,
                    point_valid_mask=batch_valid,
                    config=config,
                )
            else:
                prior = directional_shadow_prior(
                    batch_points,
                    batch_object,
                    direction.to(device),
                    receiver_mask=batch_receiver,
                    point_valid_mask=batch_valid,
                    config=config,
                )
            probability = prior[0, 0].float().cpu().clamp(0.0, 1.0)
            _atomic_npz(
                prior_path,
                schema=np.asarray(PHYSICS_CACHE_SCHEMA),
                cache_id=np.asarray(physics_cache_id),
                geometry_cache_id=np.asarray(geometry_cache_id),
                coarse_size=np.asarray(coarse_size, dtype=np.int32),
                physics_prior=probability.numpy().astype(np.float16),
                light_position_cv=light_cv.numpy().astype(np.float32),
                light_direction=direction.numpy().astype(np.float32),
                geometry_mode=np.asarray(mode),
                prior_kind=np.asarray(args.prior_kind),
                moge_scale=np.asarray(float(metadata.get("moge_scale", 1.0)), dtype=np.float32),
            )
            _write_preview(
                preview_path,
                probability,
                tuple(int(value) for value in object_mask.shape[-2:]),
            )
            generated_priors += 1
            if index % 25 == 0 or index == len(rows):
                print(
                    f"physics={index}/{len(rows)} generated={generated_priors}",
                    flush=True,
                )

    output_manifest = (
        args.output_manifest.resolve()
        if args.output_manifest
        else output_root
        / (
            f"{args.manifest.stem}_{args.geometry_mode}_geometry.jsonl"
            if args.geometry_only
            else f"{args.manifest.stem}_{args.geometry_mode}_{args.prior_kind}.jsonl"
        )
    )
    atomic_write_jsonl(output_manifest, output_rows)
    summary = {
        "schema": (
            "tokenlight_shadow_geometry_cache_run_v1"
            if args.geometry_only
            else "tokenlight_shadow_physics_cache_v1"
        ),
        "source_manifest": args.manifest.resolve().as_posix(),
        "output_manifest": output_manifest.as_posix(),
        "output_root": output_root.as_posix(),
        "rows": len(output_rows),
        "scenes": len(scene_cache_ids),
        "geometry_only": bool(args.geometry_only),
        # Backward-compatible aggregate used by the original prior-cache
        # runtime tests and existing automation.
        "generated_this_run": (
            generated_geometry if args.geometry_only else generated_priors
        ),
        "generated_geometry_this_run": generated_geometry,
        "generated_priors_this_run": generated_priors,
        "geometry_mode": args.geometry_mode,
        "prior_kind": None if args.geometry_only else args.prior_kind,
        "physics_config": None if args.geometry_only else asdict(config),
        "point_feature_scale": args.feature_scale,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
