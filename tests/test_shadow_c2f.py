from __future__ import annotations

import unittest

import torch

from model.shadow_c2f import (
    PhysicsPriorConfig,
    ShadowC2FConfig,
    ShadowCoarseToFine,
    binary_mask_metrics,
    blender_world_to_opencv_camera_matrix,
    build_adapter_features,
    directional_shadow_prior,
    point_light_shadow_prior,
    shadow_c2f_loss,
    transform_directions,
    transform_points,
    world_point_light_to_opencv_camera,
)


def _line_geometry(*, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """One occluder at x=0 with receiver points extending on both sides."""

    points = torch.zeros(1, 3, 1, 5, dtype=torch.float32)
    points[0, 0, 0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    points[0, 1, 0] = torch.tensor([0.0, 0.0, 0.0, 0.2, 0.4])
    points[0, 2, 0] = 2.0
    points.requires_grad_(requires_grad)
    object_mask = torch.zeros(1, 1, 1, 5)
    object_mask[0, 0, 0, 2] = 1.0
    return points, object_mask


def _prior_config(*, object_chunk: int = 1, receiver_chunk: int = 2) -> PhysicsPriorConfig:
    return PhysicsPriorConfig(
        coarse_size=(1, 5),
        angular_tolerance_degrees=20.0,
        temperature=0.05,
        object_chunk_size=object_chunk,
        receiver_chunk_size=receiver_chunk,
        gaussian_blur_sigma=0.0,
    )


class ShadowCoordinateTests(unittest.TestCase):
    def test_blender_to_opencv_point_and_direction(self):
        world_to_blender = torch.eye(4)
        world_to_blender[:3, 3] = torch.tensor([10.0, 20.0, 30.0])
        world_to_opencv = blender_world_to_opencv_camera_matrix(world_to_blender)
        point = transform_points(torch.tensor([1.0, 2.0, 3.0]), world_to_opencv)
        torch.testing.assert_close(point, torch.tensor([11.0, -22.0, -33.0]))

        # Translation must not affect a direction.
        direction = transform_directions(
            torch.tensor([1.0, 2.0, 3.0]),
            world_to_opencv,
            normalize=False,
        )
        torch.testing.assert_close(direction, torch.tensor([1.0, -2.0, -3.0]))

    def test_batched_transform_and_point_light_direction(self):
        transforms = torch.eye(4).repeat(2, 1, 1)
        transforms[1, 0, 3] = 2.0
        points = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
        transformed = transform_points(points, transforms)
        torch.testing.assert_close(
            transformed,
            torch.tensor([[[0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]]),
        )

        light_cv, direction_cv = world_point_light_to_opencv_camera(
            torch.tensor([1.0, 2.0, -3.0]),
            torch.zeros(3),
            torch.eye(4),
        )
        torch.testing.assert_close(light_cv, torch.tensor([1.0, -2.0, 3.0]))
        torch.testing.assert_close(direction_cv.norm(), torch.tensor(1.0))
        self.assertGreater(float(direction_cv[2]), 0.0)


class ShadowPhysicsPriorTests(unittest.TestCase):
    def test_directional_soft_prior_direction_and_gradients(self):
        points, object_mask = _line_geometry(requires_grad=True)
        light_direction = torch.tensor([-1.0, 0.0, 0.0], requires_grad=True)
        prior = directional_shadow_prior(
            points,
            object_mask,
            light_direction,
            config=_prior_config(),
            mode="soft",
        )
        self.assertEqual(tuple(prior.shape), (1, 1, 1, 5))
        self.assertGreater(float(prior[0, 0, 0, 3]), 0.5)
        self.assertGreater(float(prior[0, 0, 0, 4]), 0.5)
        self.assertLess(float(prior[0, 0, 0, 1]), 1e-5)
        self.assertEqual(float(prior[0, 0, 0, 2]), 0.0)  # object is not a receiver

        prior.sum().backward()
        self.assertIsNotNone(points.grad)
        self.assertIsNotNone(light_direction.grad)
        self.assertTrue(bool(torch.isfinite(points.grad).all()))
        self.assertTrue(bool(torch.isfinite(light_direction.grad).all()))

    def test_hard_prior_is_binary_and_non_differentiable(self):
        points, object_mask = _line_geometry(requires_grad=True)
        prior = directional_shadow_prior(
            points,
            object_mask,
            torch.tensor([-1.0, 0.0, 0.0], requires_grad=True),
            config=_prior_config(),
            mode="hard",
        )
        self.assertFalse(prior.requires_grad)
        self.assertEqual(prior[0, 0, 0].tolist(), [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_point_light_uses_divergent_object_to_receiver_flow(self):
        points, object_mask = _line_geometry(requires_grad=True)
        prior = point_light_shadow_prior(
            points,
            object_mask,
            torch.tensor([-2.0, 0.0, 2.0]),
            config=_prior_config(),
            mode="hard",
        )
        self.assertEqual(prior[0, 0, 0].tolist(), [0.0, 0.0, 0.0, 1.0, 1.0])

        light_position = torch.tensor([-2.0, 0.3, 2.0], requires_grad=True)
        soft_prior = point_light_shadow_prior(
            points,
            object_mask,
            light_position,
            config=_prior_config(),
            mode="soft",
        )
        soft_prior.sum().backward()
        self.assertIsNotNone(light_position.grad)
        self.assertGreater(float(light_position.grad.abs().sum()), 0.0)

        # A point-light position is a point, not a direction, so the coordinate
        # origin is valid and must not be rejected as a zero vector.
        origin_light = point_light_shadow_prior(
            points,
            object_mask,
            torch.zeros(3),
            config=_prior_config(),
            mode="soft",
        )
        self.assertEqual(tuple(origin_light.shape), (1, 1, 1, 5))

    def test_receiver_mask_limits_the_cast_domain(self):
        points, object_mask = _line_geometry()
        receiver_mask = torch.zeros_like(object_mask)
        receiver_mask[0, 0, 0, 3] = 1.0
        prior = directional_shadow_prior(
            points,
            object_mask,
            torch.tensor([-1.0, 0.0, 0.0]),
            receiver_mask=receiver_mask,
            config=_prior_config(),
            mode="hard",
        )
        self.assertEqual(prior[0, 0, 0].tolist(), [0.0, 0.0, 0.0, 1.0, 0.0])

        blurred_config = PhysicsPriorConfig(
            coarse_size=(1, 5),
            angular_tolerance_degrees=20.0,
            temperature=0.05,
            object_chunk_size=1,
            receiver_chunk_size=2,
            gaussian_blur_sigma=0.8,
        )
        blurred = directional_shadow_prior(
            points,
            object_mask,
            torch.tensor([-1.0, 0.0, 0.0]),
            receiver_mask=receiver_mask,
            config=blurred_config,
        )
        self.assertEqual(float(blurred[0, 0, 0, 2]), 0.0)
        self.assertEqual(float(blurred[0, 0, 0, 4]), 0.0)

    def test_chunk_sizes_are_numerically_equivalent(self):
        points, object_mask = _line_geometry()
        light = torch.tensor([-1.0, 0.0, 0.0])
        small_chunks = directional_shadow_prior(
            points,
            object_mask,
            light,
            config=_prior_config(object_chunk=1, receiver_chunk=1),
        )
        large_chunks = directional_shadow_prior(
            points,
            object_mask,
            light,
            config=_prior_config(object_chunk=64, receiver_chunk=64),
        )
        torch.testing.assert_close(small_chunks, large_chunks)

    def test_empty_object_and_invalid_receiver_are_zero(self):
        points, object_mask = _line_geometry()
        empty = directional_shadow_prior(
            points,
            torch.zeros_like(object_mask),
            torch.tensor([-1.0, 0.0, 0.0]),
            config=_prior_config(),
        )
        self.assertEqual(float(empty.sum()), 0.0)

        points[0, :, 0, 4] = torch.inf
        prior = directional_shadow_prior(
            points,
            object_mask,
            torch.tensor([-1.0, 0.0, 0.0]),
            config=_prior_config(),
            mode="hard",
        )
        self.assertEqual(float(prior[0, 0, 0, 4]), 0.0)

        cached_points, object_mask = _line_geometry()
        cached_points[0, :, 0, 4] = 0.0  # finite zero used by the MoGe cache
        point_valid = torch.ones(1, 1, 1, 5)
        point_valid[0, 0, 0, 4] = 0.0
        cached_prior = directional_shadow_prior(
            cached_points,
            object_mask,
            torch.tensor([-1.0, 0.0, 0.0]),
            point_valid_mask=point_valid,
            config=_prior_config(),
            mode="hard",
        )
        self.assertEqual(float(cached_prior[0, 0, 0, 4]), 0.0)

    def test_default_prior_resolution_is_sixty(self):
        prior = directional_shadow_prior(
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, 1, 8, 8),
            torch.tensor([-1.0, 0.0, 0.0]),
        )
        self.assertEqual(tuple(prior.shape), (1, 1, 60, 60))


class ShadowCoarseToFineTests(unittest.TestCase):
    def test_forward_shapes_zero_initial_residual_and_backward(self):
        torch.manual_seed(7)
        config = ShadowC2FConfig(
            coarse_size=(8, 8),
            coarse_base_channels=4,
            fine_channels=4,
            fine_blocks=1,
            use_receiver_mask=True,
            adapter_feature_channels=1,
        )
        model = ShadowCoarseToFine(config)
        self.assertEqual(model.stem.body[0].in_channels, 8)
        image = torch.rand(2, 3, 16, 20)
        object_mask = (torch.rand(2, 1, 16, 20) > 0.8).float()
        receiver_mask = 1.0 - object_mask
        point_map = torch.randn(2, 3, 16, 20)
        light = torch.tensor([[0.25, -0.5, 1.2], [-0.4, 0.1, 0.8]])
        coarse_prior = torch.rand(2, 1, 8, 8)
        adapter_target = torch.rand(2, 1, 16, 20)
        adapter_source = torch.rand(2, 1, 16, 20)
        adapter_features = build_adapter_features(adapter_target, adapter_source)

        outputs = model(
            image,
            object_mask,
            point_map,
            light,
            coarse_prior,
            receiver_mask=receiver_mask,
            adapter_features=adapter_features,
        )
        self.assertEqual(tuple(outputs["coarse_logits"].shape), (2, 1, 16, 20))
        self.assertEqual(tuple(outputs["logits"].shape), (2, 1, 16, 20))
        self.assertEqual(tuple(outputs["mask"].shape), (2, 1, 16, 20))
        self.assertEqual(tuple(outputs["receiver_masked_mask"].shape), (2, 1, 16, 20))
        torch.testing.assert_close(
            outputs["receiver_masked_mask"] * object_mask,
            torch.zeros_like(outputs["receiver_masked_mask"]),
        )
        torch.testing.assert_close(outputs["residual_logits"], torch.zeros_like(outputs["residual_logits"]))
        torch.testing.assert_close(outputs["logits"], outputs["coarse_logits_full"])

        target = (torch.rand(2, 1, 16, 20) > 0.75).float()
        total, components = shadow_c2f_loss(outputs, target, valid_mask=receiver_mask)
        self.assertTrue(bool(torch.isfinite(total)))
        self.assertEqual(set(components), {
            "loss",
            "fine_loss",
            "fine_bce",
            "fine_dice",
        })
        total.backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(bool(torch.isfinite(model.head.weight.grad).all()))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.0)

    def test_zero_delta_prior_is_accepted(self):
        model = ShadowCoarseToFine(
            ShadowC2FConfig(
                coarse_size=(4, 4),
                coarse_base_channels=4,
                fine_channels=4,
                fine_blocks=1,
                use_receiver_mask=True,
                adapter_feature_channels=1,
            )
        )
        outputs = model(
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, 1, 8, 8),
            torch.zeros(1, 3, 8, 8),
            torch.tensor([0.0, -1.0, 0.0]),
            torch.zeros(1, 1, 4, 4),
            receiver_mask=torch.ones(1, 1, 8, 8),
        )
        self.assertEqual(tuple(outputs["mask"].shape), (1, 1, 8, 8))

    def test_exact_light_position_is_not_direction_normalized(self):
        torch.manual_seed(19)
        model = ShadowCoarseToFine(
            ShadowC2FConfig(
                coarse_size=(4, 4),
                coarse_base_channels=4,
                fine_channels=4,
                fine_blocks=1,
            )
        ).eval()
        common = (
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, 1, 8, 8),
            torch.zeros(1, 3, 8, 8),
        )
        receiver = torch.ones(1, 1, 8, 8)
        prior = torch.zeros(1, 1, 4, 4)
        # The head is zero-initialized; emulate a trained head to test whether
        # exact light distance can affect the predicted residual.
        torch.nn.init.normal_(model.head.weight, std=0.02)
        with torch.no_grad():
            near = model(
                *common,
                torch.tensor([0.0, 0.0, 0.5]),
                prior,
                receiver_mask=receiver,
            )["coarse_logits"]
            far = model(
                *common,
                torch.tensor([0.0, 0.0, 1.5]),
                prior,
                receiver_mask=receiver,
            )["coarse_logits"]
        self.assertFalse(torch.equal(near, far))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            model(
                *common,
                torch.tensor([float("nan"), 0.0, 1.0]),
                prior,
                receiver_mask=receiver,
            )

    def test_adapter_feature_builder_and_single_probability_alias(self):
        target_probability = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]]]])
        source_probability = torch.tensor([[[[0.3, 0.4], [0.1, 0.2]]]])
        features = build_adapter_features(target_probability, source_probability)
        self.assertEqual(tuple(features.shape), (1, 3, 2, 2))
        torch.testing.assert_close(features[:, :1], target_probability)
        torch.testing.assert_close(features[:, 1:2], source_probability)
        torch.testing.assert_close(
            features[:, 2:3],
            (target_probability - source_probability).clamp_min(0.0),
        )

        torch.manual_seed(11)
        model = ShadowCoarseToFine(
            ShadowC2FConfig(
                coarse_size=(4, 4),
                coarse_base_channels=4,
                fine_channels=4,
                fine_blocks=1,
                use_receiver_mask=True,
                adapter_feature_channels=1,
            )
        ).eval()
        image = torch.zeros(1, 3, 8, 8)
        object_mask = torch.zeros(1, 1, 8, 8)
        receiver_mask = torch.ones(1, 1, 8, 8)
        point_map = torch.zeros(1, 3, 8, 8)
        light = torch.tensor([0.0, -1.0, 0.0])
        prior = torch.zeros(1, 1, 4, 4)
        probability = torch.rand(1, 1, 8, 8)
        expanded_alias = build_adapter_features(probability, torch.zeros_like(probability))
        with torch.no_grad():
            alias_output = model(
                image,
                object_mask,
                point_map,
                light,
                prior,
                probability,
                receiver_mask=receiver_mask,
            )
            feature_output = model(
                image,
                object_mask,
                point_map,
                light,
                prior,
                receiver_mask=receiver_mask,
                adapter_features=expanded_alias,
            )
        torch.testing.assert_close(alias_output["logits"], feature_output["logits"])

    def test_default_receiver_mask_is_required(self):
        model = ShadowCoarseToFine(
            ShadowC2FConfig(
                coarse_size=(4, 4),
                coarse_base_channels=4,
                fine_channels=4,
                fine_blocks=1,
            )
        )
        with self.assertRaisesRegex(ValueError, "receiver_mask is required"):
            model(
                torch.zeros(1, 3, 8, 8),
                torch.zeros(1, 1, 8, 8),
                torch.zeros(1, 3, 8, 8),
                torch.tensor([0.0, -1.0, 0.0]),
                torch.zeros(1, 1, 4, 4),
            )

    def test_receiver_gating_is_optional_but_delta_channel_is_required(self):
        model = ShadowCoarseToFine(
            ShadowC2FConfig(
                coarse_size=(4, 4),
                coarse_base_channels=4,
                fine_channels=4,
                fine_blocks=1,
                use_receiver_mask=False,
                adapter_feature_channels=1,
            )
        )
        self.assertEqual(model.stem.body[0].in_channels, 8)
        for channels in (0, 3):
            with self.assertRaisesRegex(ValueError, "one Adapter positive-delta channel"):
                ShadowC2FConfig(adapter_feature_channels=channels)
        outputs = model(
            torch.zeros(1, 3, 8, 8),
            torch.zeros(1, 1, 8, 8),
            torch.zeros(1, 3, 8, 8),
            torch.tensor([0.0, -1.0, 0.0]),
            torch.zeros(1, 1, 4, 4),
        )
        torch.testing.assert_close(outputs["receiver_masked_mask"], outputs["mask"])

    def test_perfect_mask_metrics(self):
        target = torch.tensor(
            [[[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]]]
        )
        logits = torch.where(target > 0.5, torch.full_like(target, 20.0), torch.full_like(target, -20.0))
        metrics = binary_mask_metrics(logits, target)
        for name in ("dice", "iou", "precision", "recall", "specificity", "accuracy"):
            torch.testing.assert_close(metrics[name], torch.tensor(1.0))
        torch.testing.assert_close(metrics["ber"], torch.tensor(0.0))

    def test_empty_masks_have_perfect_overlap(self):
        target = torch.zeros(2, 1, 4, 4)
        logits = torch.full_like(target, -20.0)
        metrics = binary_mask_metrics(logits, target, reduction="none")
        torch.testing.assert_close(metrics["dice"], torch.ones(2))
        torch.testing.assert_close(metrics["iou"], torch.ones(2))
        torch.testing.assert_close(metrics["ber"], torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
