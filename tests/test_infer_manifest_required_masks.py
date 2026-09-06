from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.infer_manifest import preflight_required_masks


def _args(base_path: Path, *, fallback_key: str = "") -> Namespace:
    return Namespace(
        base_path=base_path.as_posix(),
        mask_key="predicted_shadow_mask",
        mask_fallback_key=fallback_key,
        width=16,
        height=12,
        use_mask_input=True,
        tokenlight_mask_tokens=True,
    )


def _row(**values) -> dict:
    return {
        "_manifest_index": 7,
        "scene_id": "scene_test",
        "light_id": 3,
        **values,
    }


class RequiredMaskPreflightTests(unittest.TestCase):
    def test_binary_and_empty_masks_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = Image.new("L", (16, 12), 0)
            ImageDraw.Draw(binary).rectangle((4, 3, 9, 8), fill=255)
            binary.save(root / "binary.png")
            Image.new("L", (16, 12), 0).save(root / "empty.png")

            summary = preflight_required_masks(
                [
                    _row(predicted_shadow_mask="binary.png"),
                    _row(predicted_shadow_mask="empty.png", light_id=4, _manifest_index=8),
                ],
                _args(root),
            )

            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(summary["unique_mask_count"], 2)
            self.assertEqual(summary["empty_rows"], 1)
            self.assertEqual(summary["fallback_rows"], 0)
            self.assertEqual(summary["expected_size"], [16, 12])
            self.assertEqual(summary["polarity"], "0=background,255=foreground_white")

    def test_missing_primary_fails_when_fallback_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "mask fallback is disabled"):
                preflight_required_masks([_row()], _args(Path(directory)))

    def test_explicit_fallback_is_allowed_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("L", (16, 12), 0).save(root / "fallback.png")
            summary = preflight_required_masks(
                [_row(legacy_shadow_mask="fallback.png")],
                _args(root, fallback_key="legacy_shadow_mask"),
            )
            self.assertEqual(summary["primary_rows"], 0)
            self.assertEqual(summary["fallback_rows"], 1)
            self.assertEqual(summary["empty_rows"], 1)

    def test_missing_file_and_wrong_size_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("L", (15, 12), 0).save(root / "wrong_size.png")
            rows = [
                _row(predicted_shadow_mask="missing.png"),
                _row(predicted_shadow_mask="wrong_size.png", light_id=4, _manifest_index=8),
            ]
            with self.assertRaisesRegex(ValueError, "Required mask preflight failed with 2 error") as raised:
                preflight_required_masks(rows, _args(root))
            self.assertIn("missing.png", str(raised.exception))
            self.assertIn("expected size 16x12, got 15x12", str(raised.exception))

    def test_soft_or_nonbinary_mask_fails_white_foreground_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("L", (16, 12), 128).save(root / "soft.png")
            with self.assertRaisesRegex(ValueError, "0=background and 255=foreground"):
                preflight_required_masks(
                    [_row(predicted_shadow_mask="soft.png")],
                    _args(root),
                )

    def test_require_mode_demands_both_mask_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            args.use_mask_input = False
            with self.assertRaisesRegex(ValueError, "requires --use-mask-input"):
                preflight_required_masks([], args)

            args.use_mask_input = True
            args.tokenlight_mask_tokens = False
            with self.assertRaisesRegex(ValueError, "requires --tokenlight_mask_tokens"):
                preflight_required_masks([], args)


if __name__ == "__main__":
    unittest.main()
