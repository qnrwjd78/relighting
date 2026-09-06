#!/usr/bin/env python3
"""Run LiveLight Stage-1 for the 49-frame scene_002583 height-0.9 path."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path("/workspace")
MANIFEST = ROOT / "outputs/infer/relighting_external/scene_002583/height_0.9_49/trajectory_manifest.json"
SOURCE = ROOT / "data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested/scenes/scene_002583/source.png"
DEPTH = ROOT / "outputs/infer/relighting_external/scene_002583/livelight/depth.npy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3,4,5,6,7")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/infer/relighting_external/scene_002583/height_0.9_49/livelight_focal13888")
    parser.add_argument("--focal-scale", type=float, default=1.3888034269574874)
    args = parser.parse_args()
    output = args.output.resolve()
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    frames = json.loads(MANIFEST.read_text())["frames"]
    output.mkdir(parents=True, exist_ok=True)

    def worker(gpu: str, shard: list[dict]) -> None:
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "HF_HOME": "/workspace/weights/livelight/.hf_home",
            "HF_HUB_CACHE": "/workspace/weights/livelight/.hf_cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        })
        for frame in shard:
            index = frame["frame_index"]
            light = frame["livelight"]
            frame_out = output / "runs" / f"{index:03d}_position_{frame['position_id']:03d}"
            if (frame_out / "relit.png").is_file():
                continue
            frame_out.mkdir(parents=True, exist_ok=True)
            command = [
                "/workspace/conda_envs/livelight/bin/python", "inference_livelight_stage1.py",
                "--input-image", str(SOURCE), "--depth-npy", str(DEPTH),
                "--output-dir", str(frame_out), "--ckpt-dir", "/workspace/weights/livelight/LiveLight",
                "--train-config", "/workspace/scripts/livelight_stage1_scene2583.yaml",
                "--device", "cuda", "--num-inference-steps", str(args.steps), "--seed", "42",
                "--use-xformers", "--light-u", str(light["u"]), "--light-v", str(light["v"]),
                "--light-z-rel", str(light["z_rel"]), "--light-intensity", str(light["intensity"]),
                "--light-color", "1.0,1.0,1.0",
                "--focal-scale", str(args.focal_scale),
            ]
            with (frame_out / "run.log").open("w") as log:
                subprocess.run(command, cwd="/workspace/repos/LiveLight", env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"gpu={gpu} frame={index:03d} position={frame['position_id']:03d}", flush=True)

    shards = [frames[i::len(gpus)] for i in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, shard) for gpu, shard in zip(gpus, shards)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
