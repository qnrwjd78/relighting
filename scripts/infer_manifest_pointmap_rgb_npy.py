#!/usr/bin/env python3
"""RGB-source inference for point-map checkpoints using row-level NPY maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import infer_manifest as common
from scripts import infer_manifest_moge3_pointmap as point_common
from scripts import infer_manifest_pointmap_scene_cache_v2 as point_npy


def run_inference(rows, output_dir: Path, args) -> int:
    pipe, light, type_embedding, conditioner, stream_embedding = point_npy.setup_pipeline(args)
    completed = 0
    description = Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem
    for row in common.tqdm(rows, desc=description):
        prediction = common.prediction_path(output_dir, row, args)
        target = common.target_path(row, args)
        source_path = common.source_path(row, args)
        if args.skip_existing and prediction.exists():
            if args.with_gt and target and target.exists() and not common.with_gt_path(prediction).exists():
                common.save_with_gt(prediction, source_path, target)
            completed += 1
            continue
        source = common.Image.open(source_path).convert("RGB")
        points, valid = point_npy.load_point_map(row)
        current_args = argparse.Namespace(**vars(args))
        current_args.prompt = row.get(args.prompt_key) or args.prompt
        if args.seed_key:
            current_args.seed = int(row[args.seed_key])
        video = point_common.generate_one(
            pipe,
            light,
            type_embedding,
            conditioner,
            stream_embedding,
            points,
            valid,
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


def main() -> int:
    args = point_common.parse_args()
    args.manifest = common.resolve_repo(args.manifest).as_posix()
    args.weights_dir = common.resolve_repo(args.weights_dir).as_posix()
    args.checkpoint = common.resolve_repo(args.checkpoint).as_posix()
    devices = common.parse_gpu_devices(args.gpu_devices)
    if len(devices) > 1:
        raise ValueError("This RGB NPY point-map launcher accepts one GPU")
    if devices:
        args = common.apply_worker_device(devices[0], args)
    rows = common.load_rows(Path(args.manifest), int(args.limit))
    output_dir = common.resolve_repo(args.output_dir)
    common.write_snapshot(rows, output_dir, args)
    (output_dir / "pointmap_rgb_npy_inference.json").write_text(
        json.dumps(
            {
                "source_condition": "RGB PNG encoded by Wan VAE",
                "point_map_field": "point_map",
                "row_count": len(rows),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    if not args.eval_only:
        completed = run_inference(rows, output_dir, args)
        print(f"[pointmap-rgb-npy] completed={completed} output_dir={output_dir}", flush=True)
    if args.eval or args.eval_only:
        metrics = common.run_eval(rows, output_dir, args)
        metrics_output = common.resolve_repo(args.metrics_output) if args.metrics_output else output_dir / "metrics.json"
        payload = {
            "manifest": args.manifest,
            "base_path": common.data_base(args).as_posix(),
            "pred_dir": output_dir.as_posix(),
            "device": str(common.metric_device(args.metric_device)),
            "lpips_net": args.lpips_net,
            **metrics,
        }
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[pointmap-rgb-npy] wrote metrics: {metrics_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
