"""BCE and Dice objectives for coarse-to-fine shadow masks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn


def _validate_mask_pair(logits: torch.Tensor, target: torch.Tensor) -> None:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must have shape [B,1,H,W], got {tuple(logits.shape)}")
    if target.shape != logits.shape:
        raise ValueError(f"target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}")
    if not torch.is_floating_point(logits):
        raise TypeError(f"logits must be floating point, got {logits.dtype}")


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return values.mean()
    if valid_mask.shape != values.shape:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match {tuple(values.shape)}")
    weights = valid_mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def binary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute mean soft Dice loss from logits.

    Empty prediction/target pairs receive zero loss.  ``valid_mask`` can limit
    supervision to receiver pixels and must have shape ``[B,1,H,W]``.
    """

    _validate_mask_pair(logits, target)
    if smooth <= 0:
        raise ValueError(f"smooth must be positive, got {smooth}")
    probability = torch.sigmoid(logits.float())
    target = target.to(device=logits.device, dtype=torch.float32)
    weights = torch.ones_like(probability)
    if valid_mask is not None:
        if valid_mask.shape != logits.shape:
            raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match logits")
        weights = valid_mask.to(device=logits.device, dtype=torch.float32)
    dimensions = (1, 2, 3)
    intersection = (probability * target * weights).sum(dim=dimensions)
    denominator = (probability * weights).sum(dim=dimensions) + (target * weights).sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def binary_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    positive_weight: float | None = None,
) -> torch.Tensor:
    """Compute masked binary cross entropy from logits."""

    _validate_mask_pair(logits, target)
    target = target.to(device=logits.device, dtype=logits.dtype)
    pos_weight = None
    if positive_weight is not None:
        if positive_weight <= 0:
            raise ValueError(f"positive_weight must be positive, got {positive_weight}")
        pos_weight = torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    values = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return _masked_mean(values, valid_mask)


def shadow_c2f_loss(
    outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    dice_weight: float = 0.1,
    coarse_weight: float = 0.5,
    positive_weight: float | None = None,
    smooth: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute paper-style BCE+Dice supervision for both refinement stages.

    Args:
        outputs: Mapping returned by :class:`ShadowCoarseToFine`; keys
            ``"logits"`` and ``"coarse_logits"`` are required.
        target: Full-resolution cast-shadow target ``[B,1,H,W]`` in ``[0,1]``.
        valid_mask: Optional full-resolution receiver domain.
        dice_weight: Weight of Dice relative to BCE.  The paper uses ``0.1``.
        coarse_weight: Auxiliary weight applied to the coarse-stage loss.
        positive_weight: Optional BCE positive-class weight for sparse masks.
        smooth: Dice smoothing constant.

    Returns:
        ``(total_loss, components)``.  Component tensors remain attached to the
        graph so callers may log ``value.detach()`` or combine them further.
    """

    if "logits" not in outputs or "coarse_logits" not in outputs:
        raise KeyError("outputs must contain 'logits' and 'coarse_logits'")
    if dice_weight < 0 or coarse_weight < 0:
        raise ValueError("dice_weight and coarse_weight must be non-negative")
    fine_logits = outputs["logits"]
    _validate_mask_pair(fine_logits, target)
    fine_bce = binary_bce_loss(
        fine_logits,
        target,
        valid_mask=valid_mask,
        positive_weight=positive_weight,
    )
    fine_dice = binary_dice_loss(fine_logits, target, valid_mask=valid_mask, smooth=smooth)
    fine_loss = fine_bce + dice_weight * fine_dice

    # The direct 480x480 refiner exposes compatibility aliases named
    # ``coarse_logits`` but marks itself as single-stage.  Do not supervise
    # that alias a second time.
    if bool(outputs.get("single_stage", False)):
        return fine_loss, {
            "loss": fine_loss,
            "fine_loss": fine_loss,
            "fine_bce": fine_bce,
            "fine_dice": fine_dice,
        }

    coarse_logits = outputs["coarse_logits"]
    if coarse_logits.ndim != 4 or coarse_logits.shape[:2] != target.shape[:2]:
        raise ValueError(f"coarse_logits must have shape [B,1,Hc,Wc], got {tuple(coarse_logits.shape)}")
    coarse_target = F.interpolate(target.float(), size=coarse_logits.shape[-2:], mode="area")
    coarse_valid = None
    if valid_mask is not None:
        if valid_mask.shape != target.shape:
            raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match target")
        coarse_valid = F.interpolate(valid_mask.float(), size=coarse_logits.shape[-2:], mode="area")
    coarse_bce = binary_bce_loss(
        coarse_logits,
        coarse_target,
        valid_mask=coarse_valid,
        positive_weight=positive_weight,
    )
    coarse_dice = binary_dice_loss(
        coarse_logits,
        coarse_target,
        valid_mask=coarse_valid,
        smooth=smooth,
    )
    coarse_loss = coarse_bce + dice_weight * coarse_dice
    total = fine_loss + coarse_weight * coarse_loss
    return total, {
        "loss": total,
        "fine_loss": fine_loss,
        "fine_bce": fine_bce,
        "fine_dice": fine_dice,
        "coarse_loss": coarse_loss,
        "coarse_bce": coarse_bce,
        "coarse_dice": coarse_dice,
    }


class ShadowC2FLoss(nn.Module):
    """Module wrapper around :func:`shadow_c2f_loss`."""

    def __init__(
        self,
        *,
        dice_weight: float = 0.1,
        coarse_weight: float = 0.5,
        positive_weight: float | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.coarse_weight = float(coarse_weight)
        self.positive_weight = None if positive_weight is None else float(positive_weight)
        self.smooth = float(smooth)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return shadow_c2f_loss(
            outputs,
            target,
            valid_mask=valid_mask,
            dice_weight=self.dice_weight,
            coarse_weight=self.coarse_weight,
            positive_weight=self.positive_weight,
            smooth=self.smooth,
        )


__all__ = [
    "ShadowC2FLoss",
    "binary_bce_loss",
    "binary_dice_loss",
    "shadow_c2f_loss",
]
