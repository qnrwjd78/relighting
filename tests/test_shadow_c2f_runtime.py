from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from model.shadow_c2f import ShadowC2FConfig, ShadowCoarseToFine
from scripts import build_shadow_physics_cache as physics_builder
from scripts import infer_shadow_c2f as infer_script
from scripts import train_shadow_c2f as train_script
from utils.shadow_c2f_dataset import (
    ADAPTER_DELTA_CACHE_SCHEMA,
    ADAPTER_IMAGE_CACHE_SCHEMA,
    GEOMETRY_CACHE_SCHEMA,
    PHYSICS_CACHE_SCHEMA,
    ShadowC2FDataset,
)
from utils.shadow_geometry import load_and_align_moge_point_map


def _png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8)).save(path)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ShadowC2FRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        size = 8
        self.source = self.root / "source.png"
        self.object_mask = self.root / "object.png"
        self.receiver_mask = self.root / "receiver.png"
        self.gt_mask = self.root / "gt.png"
        _png(self.source, np.full((size, size, 3), 127, dtype=np.uint8))
        object_value = np.zeros((size, size), dtype=np.uint8)
        object_value[:4, :4] = 255
        receiver_value = np.full((size, size), 255, dtype=np.uint8)
        receiver_value[:4, :4] = 0
        gt_value = np.zeros((size, size), dtype=np.uint8)
        gt_value[4:6, 2:5] = 255
        _png(self.object_mask, object_value)
        _png(self.receiver_mask, receiver_value)
        _png(self.gt_mask, gt_value)

        ys, xs = np.meshgrid(
            np.linspace(-1.0, 1.0, size, dtype=np.float32),
            np.linspace(-1.0, 1.0, size, dtype=np.float32),
            indexing="ij",
        )
        points = np.stack((xs, ys, np.full_like(xs, 3.5)), axis=-1)
        self.point_map = self.root / "points.npy"
        np.save(self.point_map, points)
        self.manifest = self.root / "manifest.jsonl"
        self.row: dict[str, object] = {
            "scene_id": "scene_000001",
            "sample_name": "position_000",
            "light_id": 0,
            "light_position": [0.5, -0.5, 1.0],
            "source_image": self.source.as_posix(),
            "object_mask": self.object_mask.as_posix(),
            "receiver_mask": self.receiver_mask.as_posix(),
            "point_map": self.point_map.as_posix(),
            "gt_shadow_mask": self.gt_mask.as_posix(),
        }
        _jsonl(self.manifest, [self.row])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run_builder(self, *extra: str) -> tuple[dict[str, object], dict[str, object]]:
        output_root = self.root / "physics"
        argv = [
            "build_shadow_physics_cache.py",
            "--manifest",
            str(self.manifest),
            "--output-root",
            str(output_root),
            "--geometry-mode",
            "moge",
            "--prior-kind",
            "point",
            "--device",
            "cpu",
            "--coarse-size",
            "4",
            "--blur-sigma",
            "0",
            "--object-chunk-size",
            "4",
            "--receiver-chunk-size",
            "8",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            self.assertEqual(physics_builder.main(), 0)
        output_manifest = output_root / (
            "manifest_moge_geometry.jsonl"
            if "--geometry-only" in extra
            else "manifest_moge_point.jsonl"
        )
        row = json.loads(output_manifest.read_text(encoding="utf-8"))
        summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
        return row, summary

    def test_cpu_null_context_factory_is_a_context_manager(self) -> None:
        for precision in ("fp32", "bf16"):
            context_factory = train_script._amp_context_factory(torch.device("cpu"), precision)
            with context_factory():
                value = torch.ones(1) + 1
            self.assertEqual(value.item(), 2.0)

    def test_moge_alignment_resizes_to_mask_and_rejects_behind_camera_points(self) -> None:
        small = np.load(self.point_map, allow_pickle=False)[::2, ::2].copy()
        small[0, 0, 2] = -2.0
        path = self.root / "small.npy"
        np.save(path, small)
        object_mask = torch.ones(1, 8, 8)
        points, valid, _ = load_and_align_moge_point_map(path, object_mask)
        self.assertEqual(tuple(points.shape), (3, 8, 8))
        self.assertEqual(tuple(valid.shape), (1, 8, 8))
        self.assertTrue(torch.isfinite(points).all())
        self.assertTrue(bool((points[2:3][valid.bool()] > 0).all()))

    def test_physics_cache_schema_and_config_fingerprint(self) -> None:
        first, first_summary = self._run_builder()
        self.assertEqual(first_summary["generated_this_run"], 1)
        with np.load(first["geometry_cache"], allow_pickle=False) as archive:
            self.assertEqual(str(archive["schema"].item()), GEOMETRY_CACHE_SCHEMA)
            self.assertEqual(archive["point_features"].shape, (3, 8, 8))
        with np.load(first["physics_cache"], allow_pickle=False) as archive:
            self.assertEqual(str(archive["schema"].item()), PHYSICS_CACHE_SCHEMA)
            self.assertEqual(archive["physics_prior"].shape, (4, 4))

        second, second_summary = self._run_builder()
        self.assertEqual(second_summary["generated_this_run"], 0)
        self.assertEqual(first["physics_cache_id"], second["physics_cache_id"])

        changed, changed_summary = self._run_builder("--angular-tolerance", "6")
        self.assertEqual(changed_summary["generated_this_run"], 1)
        self.assertEqual(first["geometry_cache_id"], changed["geometry_cache_id"])
        self.assertNotEqual(first["physics_cache_id"], changed["physics_cache_id"])

    def test_geometry_only_default_contract_needs_no_physics_cache(self) -> None:
        row, summary = self._run_builder("--geometry-only")
        self.assertTrue(summary["geometry_only"])
        self.assertNotIn("physics_cache", row)
        manifest = self.root / "geometry_only.jsonl"
        _jsonl(manifest, [row])
        item = ShadowC2FDataset(
            manifest,
            image_size=8,
            adapter_mode="zero",
            coarse_prior_mode="adapter_delta",
        )[0]
        self.assertEqual(tuple(item["coarse_prior"].shape), (1, 8, 8))
        self.assertEqual(tuple(item["light_position"].shape), (3,))

    def test_dataset_target_is_optional_and_cache_ids_must_match(self) -> None:
        row, _ = self._run_builder()
        row.pop("gt_shadow_mask")
        manifest = self.root / "no_gt.jsonl"
        _jsonl(manifest, [row])
        dataset = ShadowC2FDataset(
            manifest,
            image_size=8,
            adapter_mode="zero",
            require_target=False,
        )
        item = dataset[0]
        self.assertNotIn("target", item)
        self.assertEqual(tuple(item["point_map"].shape), (3, 8, 8))
        self.assertEqual(tuple(item["coarse_prior"].shape), (1, 8, 8))
        self.assertEqual(tuple(item["light_position"].shape), (3,))
        expected_light = torch.tensor([0.5, -1.0, 3.0]) / 4.0
        torch.testing.assert_close(item["light_position"], expected_light)
        with self.assertRaisesRegex(KeyError, "gt_shadow_mask"):
            ShadowC2FDataset(manifest, image_size=8, adapter_mode="zero")[0]

        physics_path = Path(str(row["physics_cache"]))
        with np.load(physics_path, allow_pickle=False) as archive:
            values = {key: np.asarray(archive[key]).copy() for key in archive.files}
        values["geometry_cache_id"] = np.asarray("wrong-geometry")
        np.savez_compressed(physics_path, **values)
        with self.assertRaisesRegex(ValueError, "Geometry/physics cache mismatch"):
            ShadowC2FDataset(
                manifest,
                image_size=8,
                adapter_mode="zero",
                coarse_prior_mode="physics",
                require_target=False,
            )[0]

    def test_adapter_cache_id_mismatch_is_rejected(self) -> None:
        row, _ = self._run_builder()
        cache_paths = {
            "adapter_target_cache": self.root / "adapter_target.npz",
            "adapter_source_cache": self.root / "adapter_source.npz",
            "adapter_delta_cache": self.root / "adapter_delta.npz",
        }
        probability = np.full((8, 8), 0.25, dtype=np.float32)
        np.savez_compressed(
            cache_paths["adapter_target_cache"],
            schema=np.asarray(ADAPTER_IMAGE_CACHE_SCHEMA),
            cache_id=np.asarray("model-a"),
            output_size=np.asarray(8),
            final_prob=probability,
        )
        np.savez_compressed(
            cache_paths["adapter_source_cache"],
            schema=np.asarray(ADAPTER_IMAGE_CACHE_SCHEMA),
            cache_id=np.asarray("model-b"),
            output_size=np.asarray(8),
            final_prob=probability,
        )
        np.savez_compressed(
            cache_paths["adapter_delta_cache"],
            schema=np.asarray(ADAPTER_DELTA_CACHE_SCHEMA),
            cache_id=np.asarray("model-a"),
            output_size=np.asarray(8),
            positive_delta_final_prob=np.zeros_like(probability),
        )
        row.update({key: path.as_posix() for key, path in cache_paths.items()})
        manifest = self.root / "adapter.jsonl"
        _jsonl(manifest, [row])
        with self.assertRaisesRegex(ValueError, "cache_id mismatch"):
            ShadowC2FDataset(manifest, image_size=8, adapter_mode="required")[0]

    def test_adapter_delta_is_the_default_coarse_prior(self) -> None:
        row, _ = self._run_builder("--geometry-only")
        target_path = self.root / "adapter_target_ok.npz"
        source_path = self.root / "adapter_source_ok.npz"
        delta_path = self.root / "adapter_delta_ok.npz"
        target = np.full((8, 8), 0.7, dtype=np.float32)
        source = np.full((8, 8), 0.2, dtype=np.float32)
        target_coarse = np.full((8, 8), 0.8, dtype=np.float32)
        source_coarse = np.full((8, 8), 0.3, dtype=np.float32)
        np.savez_compressed(
            target_path,
            schema=np.asarray(ADAPTER_IMAGE_CACHE_SCHEMA),
            cache_id=np.asarray("adapter-ok"),
            output_size=np.asarray(8),
            final_prob=target,
            coarse_prob=target_coarse,
        )
        np.savez_compressed(
            source_path,
            schema=np.asarray(ADAPTER_IMAGE_CACHE_SCHEMA),
            cache_id=np.asarray("adapter-ok"),
            output_size=np.asarray(8),
            final_prob=source,
            coarse_prob=source_coarse,
        )
        np.savez_compressed(
            delta_path,
            schema=np.asarray(ADAPTER_DELTA_CACHE_SCHEMA),
            cache_id=np.asarray("adapter-ok"),
            output_size=np.asarray(8),
            positive_delta_final_prob=target - source,
            positive_delta_coarse_prob=target_coarse - source_coarse,
        )
        row.update(
            {
                "adapter_target_cache": target_path.as_posix(),
                "adapter_source_cache": source_path.as_posix(),
                "adapter_delta_cache": delta_path.as_posix(),
            }
        )
        manifest = self.root / "adapter_ok.jsonl"
        _jsonl(manifest, [row])
        item = ShadowC2FDataset(manifest, image_size=8)[0]
        torch.testing.assert_close(
            item["coarse_prior"],
            torch.full((1, 8, 8), 0.5),
        )
        torch.testing.assert_close(
            item["adapter_features"][2],
            torch.full((8, 8), 0.5),
        )

    def test_cpu_inference_separates_wan_and_evaluation_manifests_and_validates_cache(self) -> None:
        row, _ = self._run_builder()
        physics_manifest = self.root / "physics_manifest.jsonl"
        _jsonl(physics_manifest, [row])
        config = ShadowC2FConfig(
            coarse_size=(4, 4),
            coarse_base_channels=2,
            fine_channels=2,
            fine_blocks=1,
        )
        model = ShadowCoarseToFine(config)
        checkpoint = self.root / "checkpoint.pt"
        torch.save(
            {
                "schema": "tokenlight_shadow_c2f_checkpoint_v1",
                "threshold": 0.5,
                "coarse_prior_mode": "adapter_delta",
                "model_config": config.to_dict(),
                "model": model.state_dict(),
            },
            checkpoint,
        )
        output_root = self.root / "infer"

        def small_dataset(*args, **kwargs):
            self.assertFalse(kwargs["require_target"])
            kwargs["image_size"] = 16
            return ShadowC2FDataset(*args, **kwargs)

        def run(threshold: str = "0.5") -> dict[str, object]:
            argv = [
                "infer_shadow_c2f.py",
                "--manifest",
                str(physics_manifest),
                "--checkpoint",
                str(checkpoint),
                "--output-root",
                str(output_root),
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--adapter-mode",
                "zero",
                "--batch-size",
                "1",
                "--num-workers",
                "0",
                "--min-component-area",
                "0",
                "--threshold",
                threshold,
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                infer_script, "ShadowC2FDataset", side_effect=small_dataset
            ):
                self.assertEqual(infer_script.main(), 0)
            return json.loads((output_root / "inference_summary.json").read_text())

        first = run()
        self.assertEqual(first["validated_cache_hits"], 0)
        wan = json.loads((output_root / "wan_manifest.jsonl").read_text())
        evaluation = json.loads((output_root / "evaluation_manifest.jsonl").read_text())
        self.assertNotIn("gt_shadow_mask", wan)
        self.assertEqual(evaluation["gt_shadow_mask"], self.gt_mask.as_posix())
        first_cache_id = wan["shadow_mask_cache_id"]

        second = run()
        self.assertEqual(second["validated_cache_hits"], 1)
        changed = run("0.6")
        self.assertEqual(changed["validated_cache_hits"], 0)
        changed_wan = json.loads((output_root / "wan_manifest.jsonl").read_text())
        self.assertNotEqual(first_cache_id, changed_wan["shadow_mask_cache_id"])


if __name__ == "__main__":
    unittest.main()
