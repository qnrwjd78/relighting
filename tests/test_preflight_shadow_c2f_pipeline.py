from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load(
    "preflight_shadow_c2f_pipeline_test",
    "scripts/preflight_shadow_c2f_pipeline.py",
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _save_mask(path: Path, *, size: tuple[int, int] = (480, 480), value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(size[::-1], value, dtype=np.uint8), mode="L").save(path)


def _save_binary_mask(path: Path, *, size: tuple[int, int] = (480, 480)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0]), dtype=np.uint8)
    array[100:180, 120:220] = 255
    Image.fromarray(array, mode="L").save(path)


def _save_point_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((480, 480, 3), dtype=np.float32))


def _save_adapter_image_cache(path: Path, *, cache_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.zeros((480, 480), dtype=np.float32)
    np.savez_compressed(
        path,
        schema=np.asarray("adaptershadow_image_v1"),
        cache_id=np.asarray(cache_id),
        output_size=np.asarray(480),
        final_logit=grid,
        final_prob=grid,
        coarse_logit=grid,
        coarse_prob=grid,
    )


def _save_adapter_delta_cache(path: Path, *, cache_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.zeros((480, 480), dtype=np.float32)
    np.savez_compressed(
        path,
        schema=np.asarray("adaptershadow_positive_delta_v1"),
        cache_id=np.asarray(cache_id),
        output_size=np.asarray(480),
        positive_delta_final_prob=grid,
        positive_delta_coarse_prob=grid,
    )


def _save_geometry_cache(path: Path, *, cache_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray("tokenlight_shadow_geometry_cache_v1"),
        cache_id=np.asarray(cache_id),
        point_features=np.zeros((3, 480, 480), dtype=np.float32),
        point_valid=np.ones((1, 480, 480), dtype=np.float32),
    )


def _save_physics_cache(path: Path, *, cache_id: str, geometry_cache_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray("tokenlight_shadow_physics_prior_v1"),
        cache_id=np.asarray(cache_id),
        geometry_cache_id=np.asarray(geometry_cache_id),
        coarse_size=np.asarray([60, 60], dtype=np.int64),
        physics_prior=np.zeros((60, 60), dtype=np.float32),
        light_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )


def _save_safetensors_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except Exception:
        torch.save({"tensor": torch.ones(1)}, path)
        return
    save_file({"tensor": torch.ones(1)}, str(path))


def _save_c2f_checkpoint(path: Path, *, schema: str = "tokenlight_shadow_c2f_checkpoint_v1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": schema,
            "epoch": 1,
            "coarse_prior_mode": "adapter_delta",
            "model_config": {"coarse_size": [60, 60]},
            "model": {"weight": torch.ones(1)},
        },
        path,
    )


def _adapter_root(root: Path, *, include_checkpoint: bool = True) -> tuple[Path, Path]:
    adapter_root = root / "external" / "AdapterShadow"
    (adapter_root / "models_exp" / "sam").mkdir(parents=True, exist_ok=True)
    (adapter_root / "models_exp" / "sam" / "build_sam.py").write_text("x=1\n", encoding="utf-8")
    hub = adapter_root / "efficientnet" / "hub" / "rwightman_gen-efficientnet-pytorch_master"
    hub.mkdir(parents=True, exist_ok=True)
    (hub / "hubconf.py").write_text("x=1\n", encoding="utf-8")
    sam = adapter_root / "checkpoint" / "sam" / "sam_vit_b_01ec64.pth"
    sam.parent.mkdir(parents=True, exist_ok=True)
    sam.write_bytes(b"sam")
    efficientnet = (
        adapter_root
        / "efficientnet"
        / "hub"
        / "checkpoints"
        / "tf_efficientnet_b1_ap-44ef0a3d.pth"
    )
    efficientnet.parent.mkdir(parents=True, exist_ok=True)
    efficientnet.write_bytes(b"eff")
    checkpoint = adapter_root / "checkpoint" / "sbu.ckpt"
    if include_checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"adapter")
    return adapter_root, checkpoint


def _args(
    temp_root: Path,
    manifest: Path,
    adapter_root: Path,
    adapter_checkpoint: Path,
    *,
    c2f_checkpoint: Path | None = None,
    wan_checkpoint: Path | None = None,
    allow_incomplete: bool = False,
) -> Namespace:
    return Namespace(
        manifest=[manifest],
        baseline_checkpoint=temp_root / "baseline.safetensors",
        baseline_dir=temp_root / "baseline_predictions",
        adaptershadow_root=adapter_root,
        adapter_checkpoint=adapter_checkpoint,
        sam_checkpoint=None,
        efficientnet_checkpoint=None,
        c2f_checkpoint=c2f_checkpoint,
        wan_checkpoint=wan_checkpoint,
        base_path=temp_root,
        output_report=None,
        allow_incomplete=allow_incomplete,
    )


class ShadowC2FPreflightTests(unittest.TestCase):
    def test_complete_preflight_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_safetensors_checkpoint(root / "baseline.safetensors")
            _save_safetensors_checkpoint(root / "wan.safetensors")
            _save_c2f_checkpoint(root / "c2f.pt")
            (root / "baseline_predictions").mkdir()
            adapter_root, adapter_checkpoint = _adapter_root(root)

            rows = []
            for index, split in enumerate(("train", "val"), start=1):
                scene = f"scene_{index:06d}"
                sample = f"light_{index:03d}"
                source = root / scene / "source.png"
                baseline = root / "baseline_predictions" / f"{scene}_light_{index:03d}.png"
                source.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (480, 480), (10, 20, 30)).save(source)
                Image.new("RGB", (480, 480), (30, 20, 10)).save(baseline)
                _save_mask(root / scene / "object_mask.png", value=0)
                _save_mask(root / scene / "receiver_mask.png", value=255)
                _save_binary_mask(root / scene / "gt.png")
                _save_binary_mask(root / scene / "pred.png")
                _save_point_map(root / scene / "point.npy")
                cache_id = f"adapter-{index}"
                geometry_id = f"geometry-{index}"
                _save_adapter_image_cache(root / scene / "adapter_source.npz", cache_id=cache_id)
                _save_adapter_image_cache(root / scene / "adapter_target.npz", cache_id=cache_id)
                _save_adapter_delta_cache(root / scene / "adapter_delta.npz", cache_id=cache_id)
                _save_geometry_cache(root / scene / "geometry.npz", cache_id=geometry_id)
                _save_physics_cache(
                    root / scene / "physics.npz",
                    cache_id=f"physics-{index}",
                    geometry_cache_id=geometry_id,
                )
                rows.append(
                    {
                        "scene_id": scene,
                        "sample_name": sample,
                        "shadow_c2f_sample_id": f"{scene}__{sample}",
                        "shadow_c2f_split": split,
                        "light_position": [0.1 * index, -0.2 * index, 0.3 * index],
                        "source_image": source.relative_to(root).as_posix(),
                        "baseline_image": baseline.relative_to(root).as_posix(),
                        "object_mask": (root / scene / "object_mask.png").relative_to(root).as_posix(),
                        "receiver_mask": (root / scene / "receiver_mask.png").relative_to(root).as_posix(),
                        "gt_shadow_mask": (root / scene / "gt.png").relative_to(root).as_posix(),
                        "predicted_shadow_mask": (root / scene / "pred.png").relative_to(root).as_posix(),
                        "point_map": (root / scene / "point.npy").relative_to(root).as_posix(),
                        "adapter_source_cache": (root / scene / "adapter_source.npz").relative_to(root).as_posix(),
                        "adapter_target_cache": (root / scene / "adapter_target.npz").relative_to(root).as_posix(),
                        "adapter_delta_cache": (root / scene / "adapter_delta.npz").relative_to(root).as_posix(),
                        "geometry_cache": (root / scene / "geometry.npz").relative_to(root).as_posix(),
                        "physics_cache": (root / scene / "physics.npz").relative_to(root).as_posix(),
                    }
                )

            manifest = root / "manifest.jsonl"
            _write_jsonl(manifest, rows)
            report = MODULE.run_preflight(
                _args(
                    root,
                    manifest,
                    adapter_root,
                    adapter_checkpoint,
                    c2f_checkpoint=root / "c2f.pt",
                    wan_checkpoint=root / "wan.safetensors",
                )
            )

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["totals"]["row_count"], 2)
            self.assertEqual(report["totals"]["scene_count"], 2)
            self.assertEqual(report["masks"]["predicted_shadow_mask"]["valid_binary_480"], 2)
            self.assertEqual(report["checkpoints"]["c2f_checkpoint"]["schema"], "tokenlight_shadow_c2f_checkpoint_v1")
            self.assertEqual(MODULE.exit_code_for_report(report, allow_incomplete=False), 0)

    def test_missing_adapter_checkpoint_is_actionable_and_allow_incomplete_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_safetensors_checkpoint(root / "baseline.safetensors")
            (root / "baseline_predictions").mkdir()
            scene = "scene_000001"
            (root / scene).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (480, 480), (10, 10, 10)).save(root / scene / "source.png")
            Image.new("RGB", (480, 480), (20, 20, 20)).save(
                root / "baseline_predictions" / f"{scene}_light_001.png"
            )
            _save_mask(root / scene / "object_mask.png", value=0)
            _save_mask(root / scene / "receiver_mask.png", value=255)
            _save_binary_mask(root / scene / "gt.png")
            manifest = root / "manifest.jsonl"
            _write_jsonl(
                manifest,
                [
                    {
                        "scene_id": scene,
                        "sample_name": "light_001",
                        "shadow_c2f_split": "train",
                        "light_position": [0.0, 0.0, 1.0],
                        "source_image": f"{scene}/source.png",
                        "baseline_image": f"baseline_predictions/{scene}_light_001.png",
                        "object_mask": f"{scene}/object_mask.png",
                        "receiver_mask": f"{scene}/receiver_mask.png",
                        "gt_shadow_mask": f"{scene}/gt.png",
                        "geometry_cache": f"{scene}/missing_geometry.npz",
                    }
                ],
            )
            adapter_root, adapter_checkpoint = _adapter_root(root, include_checkpoint=False)
            report = MODULE.run_preflight(
                _args(root, manifest, adapter_root, adapter_checkpoint, allow_incomplete=True)
            )

            self.assertFalse(report["ok"])
            self.assertIn("sbu.ckpt", report["checkpoints"]["adaptershadow"]["error"])
            self.assertEqual(MODULE.exit_code_for_report(report, allow_incomplete=True), 0)
            self.assertEqual(MODULE.exit_code_for_report(report, allow_incomplete=False), 1)

    def test_gt_leakage_and_scene_split_conflict_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_safetensors_checkpoint(root / "baseline.safetensors")
            (root / "baseline_predictions").mkdir()
            adapter_root, adapter_checkpoint = _adapter_root(root)
            scene = "scene_000001"
            (root / scene).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (480, 480), (10, 10, 10)).save(root / scene / "source.png")
            Image.new("RGB", (480, 480), (20, 20, 20)).save(
                root / "baseline_predictions" / f"{scene}_light_001.png"
            )
            Image.new("RGB", (480, 480), (20, 20, 20)).save(
                root / "baseline_predictions" / f"{scene}_light_002.png"
            )
            _save_mask(root / scene / "object_mask.png", value=0)
            _save_mask(root / scene / "receiver_mask.png", value=255)
            _save_binary_mask(root / scene / "gt.png")
            manifest = root / "manifest.jsonl"
            _write_jsonl(
                manifest,
                [
                    {
                        "scene_id": scene,
                        "sample_name": "light_001",
                        "shadow_c2f_split": "train",
                        "light_position": [0.0, 0.0, 1.0],
                        "source_image": f"{scene}/source.png",
                        "baseline_image": f"baseline_predictions/{scene}_light_001.png",
                        "object_mask": f"{scene}/object_mask.png",
                        "receiver_mask": f"{scene}/receiver_mask.png",
                        "gt_shadow_mask": f"{scene}/gt.png",
                        "geometry_cache": f"{scene}/gt.png",
                        "shadow_mask": f"{scene}/gt.png",
                    },
                    {
                        "scene_id": scene,
                        "sample_name": "light_002",
                        "shadow_c2f_split": "val",
                        "light_position": [0.0, 0.0, 1.0],
                        "source_image": f"{scene}/source.png",
                        "baseline_image": f"baseline_predictions/{scene}_light_002.png",
                        "object_mask": f"{scene}/object_mask.png",
                        "receiver_mask": f"{scene}/receiver_mask.png",
                        "gt_shadow_mask": f"{scene}/gt.png",
                        "geometry_cache": f"{scene}/gt.png",
                    },
                ],
            )

            report = MODULE.run_preflight(_args(root, manifest, adapter_root, adapter_checkpoint))

            self.assertFalse(report["ok"])
            errors = "\n".join(report["errors"])
            self.assertIn("leaks GT shadow fields", errors)
            self.assertIn("scene/split conflicts", errors)

    def test_invalid_predicted_mask_and_c2f_schema_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_safetensors_checkpoint(root / "baseline.safetensors")
            (root / "baseline_predictions").mkdir()
            adapter_root, adapter_checkpoint = _adapter_root(root)
            scene = "scene_000001"
            (root / scene).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (480, 480), (10, 10, 10)).save(root / scene / "source.png")
            Image.new("RGB", (480, 480), (20, 20, 20)).save(
                root / "baseline_predictions" / f"{scene}_light_001.png"
            )
            _save_mask(root / scene / "object_mask.png", value=0)
            _save_mask(root / scene / "receiver_mask.png", value=255)
            _save_binary_mask(root / scene / "gt.png")
            _save_c2f_checkpoint(root / "bad_c2f.pt", schema="wrong_schema")
            Image.fromarray(np.full((480, 480), 128, dtype=np.uint8), mode="L").save(
                root / scene / "pred.png"
            )
            manifest = root / "manifest.jsonl"
            _write_jsonl(
                manifest,
                [
                    {
                        "scene_id": scene,
                        "sample_name": "light_001",
                        "shadow_c2f_split": "train",
                        "light_position": [0.0, "bad", 1.0],
                        "source_image": f"{scene}/source.png",
                        "baseline_image": f"baseline_predictions/{scene}_light_001.png",
                        "object_mask": f"{scene}/object_mask.png",
                        "receiver_mask": f"{scene}/receiver_mask.png",
                        "gt_shadow_mask": f"{scene}/gt.png",
                        "predicted_shadow_mask": f"{scene}/pred.png",
                        "geometry_cache": f"{scene}/gt.png",
                    }
                ],
            )
            report = MODULE.run_preflight(
                _args(
                    root,
                    manifest,
                    adapter_root,
                    adapter_checkpoint,
                    c2f_checkpoint=root / "bad_c2f.pt",
                )
            )

            self.assertFalse(report["ok"])
            coverage = report["coverage"]["predicted_shadow_mask"]
            self.assertEqual(coverage["invalid"], 1)
            self.assertEqual(report["light_position"]["invalid"], 1)
            self.assertIn("Unexpected C2F checkpoint schema", report["checkpoints"]["c2f_checkpoint"]["error"])


if __name__ == "__main__":
    unittest.main()
