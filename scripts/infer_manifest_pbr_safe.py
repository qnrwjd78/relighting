#!/usr/bin/env python3
"""Run PBR inference with the isolated safe per-stream timestep model.

This wrapper deliberately leaves ``scripts/infer_manifest_pbr.py`` unchanged.
It reuses that script's manifest/I/O pipeline and replaces only its runtime PBR
model function with :mod:`model.tokenlight_wan_pbr_safe`, matching checkpoints
trained by ``model/train_tokenlight_pbr_safe.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.tokenlight_wan_pbr_safe import (  # noqa: E402
    TokenLightPBRTypeEmbedding,
    model_fn_wan_video_tokenlight_pbr,
    tokenlight_pbr_type_count,
)
from scripts import infer_manifest_pbr as legacy  # noqa: E402


_legacy_runtime_imports = legacy.ensure_runtime_imports


def _safe_runtime_imports(*, include_model: bool) -> None:
    _legacy_runtime_imports(include_model=include_model)
    if include_model:
        legacy.TokenLightPBRTypeEmbedding = TokenLightPBRTypeEmbedding
        legacy.model_fn_wan_video_tokenlight_pbr = model_fn_wan_video_tokenlight_pbr
        legacy.tokenlight_pbr_type_count = tokenlight_pbr_type_count


def _install_safe_model() -> None:
    legacy.ensure_runtime_imports = _safe_runtime_imports
    legacy.TokenLightPBRTypeEmbedding = TokenLightPBRTypeEmbedding
    legacy.model_fn_wan_video_tokenlight_pbr = model_fn_wan_video_tokenlight_pbr
    legacy.tokenlight_pbr_type_count = tokenlight_pbr_type_count


def run_inference_worker_safe(
    device_id: str,
    rows: list[dict[str, Any]],
    output_dir: str,
    args_dict: dict[str, Any],
) -> int:
    """Spawn-safe worker that reinstalls the isolated model function."""

    _install_safe_model()
    args = legacy.base.apply_worker_device(
        device_id, argparse.Namespace(**args_dict)
    )
    print(
        f"[infer_manifest_pbr_safe:{device_id}] rows={len(rows)} device={args.device}",
        flush=True,
    )
    completed = legacy.run_inference(rows, Path(output_dir), args)
    print(
        f"[infer_manifest_pbr_safe:{device_id}] completed={completed}", flush=True
    )
    return int(completed)


def run_inference_distributed_safe(
    rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace
) -> int:
    devices = legacy.base.parse_gpu_devices(args.gpu_devices)
    if not devices:
        return int(legacy.run_inference(rows, output_dir, args))
    assigned = legacy.base.split_rows_by_device(rows, devices)
    if len(assigned) == 1:
        device_id, shard = assigned[0]
        worker_args = legacy.base.apply_worker_device(device_id, args)
        return int(legacy.run_inference(shard, output_dir, worker_args))

    context = mp.get_context("spawn")
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(assigned), mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                run_inference_worker_safe,
                device_id,
                shard,
                output_dir.as_posix(),
                vars(args),
            )
            for device_id, shard in assigned
        ]
        for future in concurrent.futures.as_completed(futures):
            completed += int(future.result())
    return completed


def main() -> int:
    _install_safe_model()
    legacy.run_inference_distributed = run_inference_distributed_safe
    print(
        "[infer_manifest_pbr_safe] using clean-t0 condition / sampled-t target modulation",
        flush=True,
    )
    return int(legacy.main())


if __name__ == "__main__":
    raise SystemExit(main())
