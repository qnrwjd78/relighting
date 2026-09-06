from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from model.coshadow_box_predictor import (
    CoShadowBoxPredictor,
    CoShadowBoxPredictorConfig,
    coshadow_box_loss,
    load_coshadow_box_predictor,
    quantize_boxes_xyxy,
)
from model.tokenlight_wan import _clean_prefix_t_mod
from model.tokenlight_wan_coshadow import (
    CoShadowLayoutTokenEmbedding,
    _actual_bbox_cross_attention_probability,
    _prefix_and_target_t_mod,
)
from model.train_tokenlight_coshadow import _validated_bbox_fields
from scripts.build_fixed32_coshadow_metadata import (
    foreground_bbox_from_mask,
    quantize_bbox,
    scene_split,
)


def _attrs(batch: int):
    value = {
        "a": 0.5,
        "dg": 0.0,
        "t": 1.0,
        "lights": [
            {
                "x": -0.5,
                "y": 0.25,
                "z": 1.0,
                "r": 1.0,
                "g": 1.0,
                "b": 1.0,
                "lambda": 0.3,
                "d": 0.06,
            }
        ],
    }
    return [value for _ in range(batch)]


class _DummyDit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_dim = 8
        self.dim = 4
        self.time_embedding = torch.nn.Linear(8, 8)
        self.time_projection = torch.nn.Linear(8, 24)


class CoShadowTests(unittest.TestCase):
    def test_mask_bbox_empty_edge_and_quantization(self):
        empty = Image.new("L", (10, 20), 0)
        bbox, valid, area = foreground_bbox_from_mask(empty)
        self.assertEqual(bbox, [0.0, 0.0, 0.0, 0.0])
        self.assertFalse(valid)
        self.assertEqual(area, 0)

        mask = Image.new("L", (10, 20), 0)
        ImageDraw.Draw(mask).rectangle((5, 10, 9, 19), fill=255)
        bbox, valid, area = foreground_bbox_from_mask(mask)
        self.assertTrue(valid)
        self.assertEqual(area, 50)
        self.assertEqual(bbox, [0.5, 0.5, 1.0, 1.0])
        self.assertEqual(quantize_bbox(bbox, bins=16), [8, 8, 15, 15])

    def test_scene_split_is_deterministic_and_exclusive(self):
        ratios = {"train": 0.9, "val": 0.05, "test": 0.05}
        assignments = {
            scene_split(f"scene_{index:06d}", seed=7, ratios=ratios)
            for index in range(1000)
        }
        self.assertEqual(assignments, {"train", "val", "test"})
        self.assertEqual(
            scene_split("scene_000001", seed=7, ratios=ratios),
            scene_split("scene_000001", seed=7, ratios=ratios),
        )

    def test_box_predictor_coordconv_shapes_and_loss(self):
        config = CoShadowBoxPredictorConfig(
            visual_channels=(8, 16),
            light_dim=16,
            hidden_dim=32,
            fourier_features=4,
            use_coordconv=True,
        )
        model = CoShadowBoxPredictor(config)
        self.assertEqual(model.visual_encoder[0][0].in_channels, 6)
        source = torch.randn(2, 3, 32, 32)
        mask = torch.rand(2, 1, 32, 32)
        outputs = model(source, mask, _attrs(2))
        self.assertEqual(tuple(outputs["boxes"].shape), (2, 4))
        self.assertEqual(tuple(outputs["presence_logits"].shape), (2,))
        self.assertTrue(bool(torch.all(outputs["boxes"][:, 2:] >= outputs["boxes"][:, :2])))
        target = torch.tensor([[0.1, 0.2, 0.8, 0.9], [0.0, 0.0, 0.0, 0.0]])
        loss, metrics = coshadow_box_loss(outputs, target, torch.tensor([True, False]))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(metrics["mean_iou"])))

    def test_box_predictor_checkpoint_schema_roundtrip(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = CoShadowBoxPredictorConfig(
                visual_channels=(8,),
                light_dim=8,
                hidden_dim=16,
                fourier_features=4,
                use_coordconv=True,
            )
            model = CoShadowBoxPredictor(config)
            save_file(model.state_dict(), str(path / "model.safetensors"))
            (path / "config.json").write_text(
                json.dumps({"schema": "coshadow_box_predictor_v1", "model": config.to_dict()}),
                encoding="utf-8",
            )
            loaded = load_coshadow_box_predictor(path)
            self.assertTrue(loaded.config.use_coordconv)
            self.assertEqual(loaded.visual_encoder[0][0].in_channels, 6)

    def test_layout_embedding_and_quantization(self):
        module = CoShadowLayoutTokenEmbedding(12, bins=16)
        boxes = torch.tensor([[0.0, 0.5, 1.0, 0.25], [0.1, 0.2, 0.3, 0.4]])
        bins = quantize_boxes_xyxy(boxes, bins=16)
        tokens = module(bins, torch.tensor([True, False]))
        self.assertEqual(tuple(tokens.shape), (2, 4, 12))
        self.assertEqual(bins[0].tolist(), [0, 8, 15, 4])
        with self.assertRaises(ValueError):
            module(torch.tensor([[16, 0, 0, 0]]), True)
        half_bin = quantize_boxes_xyxy(torch.tensor([[0.3, 0.3, 0.3, 0.3]]), bins=16)
        self.assertEqual(half_bin[0].tolist(), [5, 5, 5, 5])

    def test_prefix_timestep_promotion_batch_two(self):
        dit = _DummyDit()
        sampled = torch.randn(1, 6, 4)
        promoted = _prefix_and_target_t_mod(
            dit,
            sampled,
            prefix_len=5,
            target_len=7,
            batch=2,
        )
        self.assertEqual(tuple(promoted.shape), (2, 12, 6, 4))
        expected_clean = _clean_prefix_t_mod(dit, 5, 2, sampled.dtype, sampled.device)
        torch.testing.assert_close(promoted[:, :5], expected_clean)
        torch.testing.assert_close(
            promoted[:, 5:],
            sampled.expand(2, -1, -1)[:, None].expand(2, 7, 6, 4),
        )

    def test_actual_cross_attention_alignment_shapes(self):
        class Cross(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads, self.head_dim, self.has_image_input = 3, 4, False
                self.q = torch.nn.Linear(12, 12)
                self.k = torch.nn.Linear(12, 12)
                self.norm_q = torch.nn.Identity()
                self.norm_k = torch.nn.Identity()

        probabilities = _actual_bbox_cross_attention_probability(
            Cross(), torch.randn(2, 9, 12), torch.randn(2, 10, 12), query_chunk_size=4
        )
        self.assertEqual(tuple(probabilities.shape), (2, 9))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))

    def test_validation_bbox_contract(self):
        args = Namespace(
            coshadow_bbox_bins_key="coshadow_bbox_bins",
            coshadow_bbox_valid_key="coshadow_bbox_valid",
            coshadow_bbox_bins=16,
        )
        row = {
            "coshadow_bbox_xyxy": [0.0, 0.5, 1.0, 0.25],
            "coshadow_bbox_valid": True,
            "coshadow_bbox_bins": [0, 8, 15, 4],
            "coshadow_bbox_num_bins": 16,
        }
        with self.assertRaisesRegex(ValueError, "positive width and height"):
            _validated_bbox_fields(row, args, where="test:1")
        row["coshadow_bbox_xyxy"] = [0.0, 0.25, 1.0, 0.5]
        row["coshadow_bbox_bins"] = [0, 4, 15, 8]
        self.assertTrue(_validated_bbox_fields(row, args, where="test:1"))


if __name__ == "__main__":
    unittest.main()
