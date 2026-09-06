#!/usr/bin/env python3
"""TokenLight inference using cached source latents per scene.

The output/eval naming contract matches ``scripts/infer_manifest.py``.  Only
the source encoding path is replaced, so cached RGB or luminance checkpoints
are not fed a newly encoded source by mistake.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import infer as infer_core
from scripts import infer_manifest as common


def custom_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene-cache-root", required=True)
    parser.add_argument(
        "--scene-cache-transform",
        choices=("luminance", "rgb"),
        default="luminance",
        help="Expected image_transform stored in each scene cache.",
    )
    return parser.parse_known_args(argv)


class SceneSourceLatents:
    def __init__(self, root: Path, *, expected_transform: str = "luminance") -> None:
        self.root = root.resolve()
        self.expected_transform = str(expected_transform)
        self.cache_path: Path | None = None
        self.latents: torch.Tensor | None = None

    def _path(self, row: dict[str, Any]) -> Path:
        value = row.get("_scene_cache_file")
        if value not in (None, ""):
            path = Path(str(value))
            return (path if path.is_absolute() else self.root / path).resolve()
        return (self.root / "scenes" / f"{row['scene_id']}.pt").resolve()

    def get(self, row: dict[str, Any]) -> torch.Tensor:
        scene_id = str(row["scene_id"])
        path = self._path(row)
        if self.cache_path != path:
            cache = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if cache.get("schema") != "wan_vae_scene_latent_cache_v2":
                raise ValueError(f"Unexpected scene cache schema: {path}")
            if cache.get("image_transform") != self.expected_transform:
                raise ValueError(
                    f"Expected {self.expected_transform} source cache, got "
                    f"{cache.get('image_transform')!r}: {path}"
                )
            latents = cache.get("source_latent")
            if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != (48, 1, 30, 30):
                raise ValueError(f"Malformed source_latent in {path}")
            if str(cache.get("scene_id")) != scene_id:
                raise ValueError(
                    f"Manifest scene {scene_id!r} does not match cache "
                    f"{cache.get('scene_id')!r}: {path}"
                )
            self.cache_path = path
            self.latents = latents.detach().cpu().contiguous()
        assert self.latents is not None
        return self.latents


def run_inference(
    rows: list[dict[str, Any]],
    output_dir: Path,
    args,
    cache_root: Path,
    cache_transform: str,
) -> int:
    common.ensure_runtime_imports(include_model=True)
    pipe, light_encoder, type_embedding = common.setup_pipeline(args)
    source_cache = SceneSourceLatents(cache_root, expected_transform=cache_transform)
    completed = 0
    description = Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem
    for row in tqdm(rows, desc=description):
        prediction = common.prediction_path(output_dir, row, args)
        target = common.target_path(row, args)
        source_path = common.source_path(row, args)
        if args.skip_existing and prediction.exists():
            if args.with_gt and target and target.exists() and not common.with_gt_path(prediction).exists():
                common.save_with_gt(prediction, source_path, target)
            completed += 1
            continue

        source_latents = source_cache.get(row)

        def cached_source_latents(current_pipe, image, current_args):
            del image, current_args
            return source_latents.unsqueeze(0).to(
                device=current_pipe.device,
                dtype=current_pipe.torch_dtype,
            )

        infer_core.encode_image_latents = cached_source_latents
        source = Image.open(source_path).convert("RGB")
        current_args = argparse.Namespace(**vars(args))
        current_args.prompt = row.get(args.prompt_key) or args.prompt
        video = common.generate(
            pipe,
            light_encoder,
            type_embedding,
            common.attrs_from_row(row, args.attrs_key),
            source,
            None,
            current_args,
            extra_masks=None,
        )
        prediction.parent.mkdir(parents=True, exist_ok=True)
        video[0].save(prediction)
        if args.with_gt and target and target.exists():
            common.save_with_gt(prediction, source, target)
        completed += 1
    return completed


def run_inference_worker(
    device_id: str,
    rows: list[dict[str, Any]],
    output_dir: str,
    args_dict: dict[str, Any],
    cache_root: str,
    cache_transform: str,
) -> int:
    args = argparse.Namespace(**args_dict)
    args.device = "cpu" if device_id == "cpu" else f"cuda:{device_id}"
    print(f"[scene-cache:{device_id}] rows={len(rows)} device={args.device}", flush=True)
    completed = run_inference(
        rows,
        Path(output_dir),
        args,
        Path(cache_root),
        cache_transform,
    )
    print(f"[scene-cache:{device_id}] completed={completed}", flush=True)
    return completed


def main() -> int:
    custom, remaining = custom_args(sys.argv[1:])
    original = sys.argv
    sys.argv = [original[0], *remaining]
    try:
        args = common.parse_args()
    finally:
        sys.argv = original
    if args.eval_only:
        raise ValueError("Use the Objaverse-245 postprocessor for metrics")
    devices = common.parse_gpu_devices(args.gpu_devices)
    rows = common.load_rows(common.resolve_repo(args.manifest), args.limit)
    output_dir = common.resolve_repo(args.output_dir)
    common.write_snapshot(rows, output_dir, args)
    snapshot = {
        "scene_cache_root": Path(custom.scene_cache_root).resolve().as_posix(),
        "scene_cache_transform": custom.scene_cache_transform,
        "source_condition": f"cached {custom.scene_cache_transform} source_latent",
    }
    (output_dir / "scene_cache_inference.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cache_root = Path(custom.scene_cache_root).resolve()
    if len(devices) <= 1:
        if devices:
            args = common.apply_worker_device(devices[0], args)
        completed = run_inference(
            rows,
            output_dir,
            args,
            cache_root,
            custom.scene_cache_transform,
        )
    else:
        assigned = common.split_rows_by_device(rows, devices)
        print(
            f"[scene-cache] gpu_devices={','.join(devices)} "
            f"processes={len(assigned)} rows={len(rows)}",
            flush=True,
        )
        completed = 0
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(assigned),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    run_inference_worker,
                    device_id,
                    shard,
                    output_dir.as_posix(),
                    vars(args),
                    cache_root.as_posix(),
                    custom.scene_cache_transform,
                )
                for device_id, shard in assigned
            ]
            for future in concurrent.futures.as_completed(futures):
                completed += int(future.result())
    print(f"completed={completed} output_dir={output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
