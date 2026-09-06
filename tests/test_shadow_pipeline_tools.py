from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load("prepare_shadow_c2f_manifest_test", "scripts/prepare_shadow_c2f_manifest.py")
mask_eval = _load("evaluate_shadow_c2f_test", "utils/evaluate_shadow_c2f.py")
rgb_eval = _load(
    "evaluate_shadow_conditioned_rgb_test",
    "utils/evaluate_shadow_conditioned_rgb.py",
)


class ShadowPipelineToolTests(unittest.TestCase):
    def test_compact_scene_selection_does_not_bias_split_hash(self):
        rows = [
            {
                "scene_id": f"scene_{index:06d}",
                "sample_name": "position_000",
                "light_position": [0.0, 0.0, 0.1],
            }
            for index in range(1000)
        ]
        selected = prepare._select_rows(
            rows,
            max_scenes=160,
            max_per_scene=1,
            seed=260831,
        )
        splits = {
            prepare._assign_split(
                row["scene_id"],
                seed=260831,
                ratios=(0.8, 0.1, 0.1),
            )
            for row in selected
        }
        self.assertEqual(len(selected), 160)
        self.assertEqual(splits, {"train", "val", "test"})

    def test_mask_metrics_support_is_applied_symmetrically(self):
        gt = np.array([[True, True], [False, False]])
        pred = np.array([[True, False], [False, False]])
        support = np.array([[True, False], [True, True]])
        values = mask_eval._metrics(pred & support, gt & support, boundary_radius=1)
        self.assertEqual(values["dice"], 1.0)
        self.assertEqual(values["iou"], 1.0)

    def test_rgb_metrics_omit_lpips_when_disabled_and_serialize_empty_region(self):
        metric = rgb_eval.Lpips(False, torch.device("cpu"), "alex")
        image = torch.zeros(1, 3, 4, 4)
        populated = rgb_eval._region_metrics(
            image,
            image,
            np.ones((4, 4), dtype=bool),
            metric,
        )
        empty = rgb_eval._region_metrics(
            image,
            image,
            np.zeros((4, 4), dtype=bool),
            metric,
        )
        self.assertNotIn("lpips", populated)
        self.assertNotIn("lpips", empty)
        self.assertEqual(populated["psnr"], 100.0)
        self.assertIsNone(rgb_eval._finite_mean([empty["psnr"]]))


if __name__ == "__main__":
    unittest.main()
