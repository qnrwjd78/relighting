from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.infer_exp0 import (
    make_model_comparison_frames,
    ordered_video_rows,
    prediction_name,
    validate_grid_rows,
)


def _row(scene: str, variant: int, cell: tuple[int, int, int], light_id: int) -> dict:
    x, y, z = cell
    return {
        "scene_id": scene,
        "light_id": light_id,
        "original_light_id": light_id,
        "light_variant_id": variant,
        "grid_cell": list(cell),
        "grid_resolution": [2, 2, 2],
        "power_scale": 0.3 + 0.6 * variant,
        "attrs_json": json.dumps(
            {
                "lights": [
                    {
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "lambda": 0.3 + 0.6 * variant,
                    }
                ]
            }
        ),
    }


def _complete_rows() -> list[dict]:
    rows = []
    light_id = 0
    for x in range(2):
        for y in range(2):
            for z in range(2):
                for variant in range(2):
                    rows.append(_row("scene_test", variant, (x, y, z), light_id))
                    light_id += 1
    return rows


class FixedGridInferenceTests(unittest.TestCase):
    def test_complete_two_variant_grid(self):
        summary = validate_grid_rows(_complete_rows())
        self.assertIsNotNone(summary)
        self.assertEqual(summary["scenes"], 1)
        self.assertEqual(summary["sequences"], 2)
        self.assertEqual(summary["rows"], 16)

    def test_missing_grid_cell_fails(self):
        rows = _complete_rows()
        rows.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete grid"):
            validate_grid_rows(rows)

    def test_duplicate_grid_cell_fails(self):
        rows = _complete_rows()
        duplicate = dict(rows[0])
        duplicate["light_id"] = 999
        rows.append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicates=1"):
            validate_grid_rows(rows)

    def test_auto_order_is_z_then_x_then_y(self):
        rows = [row for row in reversed(_complete_rows()) if row["light_variant_id"] == 0]
        ordered = ordered_video_rows(rows, "auto")
        cells = [tuple(row["grid_cell"]) for row in ordered]
        self.assertEqual(
            cells,
            [
                (0, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 0, 1),
                (0, 1, 1),
                (1, 0, 1),
                (1, 1, 1),
            ],
        )

    def test_four_model_comparison_matches_legacy_panel_shape(self):
        row = _row("scene_test", 0, (0, 0, 0), 0)
        row["video"] = "gt.png"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (16, 16), (100, 100, 100)).save(root / "gt.png")
            run_names = [
                "rgb_baseline_test",
                "rgb_decoder_loss_test",
                "rgb_shadow_mask_test",
                "rgb_delta_flow_lambda0p2_test",
            ]
            jobs = []
            for index, run_name in enumerate(run_names):
                run_dir = root / "train" / run_name
                checkpoint = run_dir / "epoch-0.safetensors"
                prediction_dir = root / "infer" / run_name / "epoch-0" / "predictions"
                prediction_dir.mkdir(parents=True)
                Image.new("RGB", (16, 16), (index * 20, 0, 0)).save(
                    prediction_dir / prediction_name(row)
                )
                jobs.append((run_dir, checkpoint))

            comparison_dir = root / "comparison"
            suffix = make_model_comparison_frames(
                [row], jobs, root / "infer", root, comparison_dir
            )
            self.assertEqual(suffix, "_4outputs_gt_5panel.png")
            panel_path = comparison_dir / f"{Path(prediction_name(row)).stem}{suffix}"
            with Image.open(panel_path) as panel:
                self.assertEqual(panel.size, (16 * 5, 16 + 35))
            config = json.loads((comparison_dir / "comparison_config.json").read_text())
            self.assertEqual(
                config["layout"],
                "baseline | decoder loss | shadow mask | delta flow lambda=0.2 | gt",
            )


if __name__ == "__main__":
    unittest.main()
