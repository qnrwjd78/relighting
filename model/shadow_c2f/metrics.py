"""Basic binary shadow-mask metrics."""

from __future__ import annotations

from typing import Literal

import torch


MetricReduction = Literal["mean", "none"]


def _safe_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    empty_value: float,
) -> torch.Tensor:
    ratio = numerator / denominator.clamp_min(1.0)
    return torch.where(denominator > 0, ratio, torch.full_like(ratio, empty_value))


@torch.no_grad()
def binary_mask_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
    valid_mask: torch.Tensor | None = None,
    reduction: MetricReduction = "mean",
) -> dict[str, torch.Tensor]:
    """Compute Dice, IoU, precision, recall, accuracy, and BER.

    Args:
        prediction: Logits or probabilities with shape ``[B,1,H,W]``.
        target: Binary/soft target with the same shape; values >= 0.5 are
            treated as shadow.
        threshold: Probability decision threshold.  When ``from_logits=True``
            it is converted to its equivalent logit threshold.
        from_logits: Whether ``prediction`` contains logits.
        valid_mask: Optional receiver domain with the same shape.
        reduction: ``"none"`` returns one value per sample; ``"mean"`` returns
            scalar batch means.

    BER is ``0.5 * (false_negative_rate + false_positive_rate)`` in ``[0,1]``.
    Empty prediction/target pairs receive Dice and IoU of one.
    """

    if prediction.ndim != 4 or prediction.shape[1] != 1 or prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have matching shape [B,1,H,W], got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if not torch.is_floating_point(prediction):
        raise TypeError(f"prediction must be floating point, got {prediction.dtype}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0,1), got {threshold}")
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    if valid_mask is not None and valid_mask.shape != prediction.shape:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match prediction")

    if from_logits:
        logit_threshold = torch.logit(
            torch.as_tensor(threshold, device=prediction.device, dtype=prediction.dtype)
        )
        predicted = prediction >= logit_threshold
    else:
        predicted = prediction >= threshold
    expected = target >= 0.5
    if valid_mask is None:
        valid = torch.ones_like(predicted, dtype=torch.bool)
    else:
        valid = valid_mask >= 0.5
    predicted = predicted & valid
    expected = expected & valid

    dimensions = (1, 2, 3)
    true_positive = (predicted & expected).sum(dim=dimensions).float()
    false_positive = (predicted & ~expected & valid).sum(dim=dimensions).float()
    false_negative = (~predicted & expected & valid).sum(dim=dimensions).float()
    true_negative = (~predicted & ~expected & valid).sum(dim=dimensions).float()
    predicted_positive = true_positive + false_positive
    target_positive = true_positive + false_negative
    target_negative = true_negative + false_positive
    valid_count = target_positive + target_negative

    dice_denominator = predicted_positive + target_positive
    union = true_positive + false_positive + false_negative
    dice = _safe_ratio(2.0 * true_positive, dice_denominator, empty_value=1.0)
    iou = _safe_ratio(true_positive, union, empty_value=1.0)
    precision = _safe_ratio(true_positive, predicted_positive, empty_value=0.0)
    # If both masks are empty, precision and recall are conventionally perfect.
    both_empty = (predicted_positive == 0) & (target_positive == 0)
    precision = torch.where(both_empty, torch.ones_like(precision), precision)
    recall = _safe_ratio(true_positive, target_positive, empty_value=1.0)
    specificity = _safe_ratio(true_negative, target_negative, empty_value=1.0)
    accuracy = _safe_ratio(true_positive + true_negative, valid_count, empty_value=1.0)
    balanced_error_rate = 0.5 * ((1.0 - recall) + (1.0 - specificity))

    metrics = {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "ber": balanced_error_rate,
    }
    if reduction == "mean":
        return {name: value.mean() for name, value in metrics.items()}
    return metrics


__all__ = ["MetricReduction", "binary_mask_metrics"]
