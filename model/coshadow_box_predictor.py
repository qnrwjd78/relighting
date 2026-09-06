"""Light-conditioned shadow-box predictor for the fixed32 CoShadow adaptation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from model.lightoken_encoder import LightokenEncoder


@dataclass(frozen=True)
class CoShadowBoxPredictorConfig:
    visual_channels: tuple[int, ...] = (32, 64, 128, 256)
    light_dim: int = 128
    hidden_dim: int = 256
    fourier_features: int = 64
    fourier_sigma: float = 5.0
    max_lights: int = 1
    dropout: float = 0.1
    use_coordconv: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CoShadowBoxPredictorConfig":
        value = dict(value or {})
        if "visual_channels" in value:
            value["visual_channels"] = tuple(int(item) for item in value["visual_channels"])
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown CoShadowBoxPredictorConfig key(s): {', '.join(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["visual_channels"] = list(self.visual_channels)
        return result


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = max(1, min(16, out_channels // 8))
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )


class CoShadowBoxPredictor(nn.Module):
    """Predict normalized XYXY shadow bounds and a shadow-presence logit.

    MultiShadow predicts from a shadow-free composite plus object mask.  The
    fixed32 dataset reuses one source for many target lights, so this adaptation
    additionally encodes the row's Lightoken attributes to make the mapping
    identifiable.
    """

    def __init__(self, config: CoShadowBoxPredictorConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, CoShadowBoxPredictorConfig)
            else CoShadowBoxPredictorConfig.from_mapping(config)
        )
        if not self.config.visual_channels:
            raise ValueError("visual_channels cannot be empty")
        blocks: list[nn.Module] = []
        in_channels = 6 if self.config.use_coordconv else 4
        for out_channels in self.config.visual_channels:
            blocks.append(_ConvBlock(in_channels, int(out_channels)))
            in_channels = int(out_channels)
        self.visual_encoder = nn.Sequential(*blocks)
        self.visual_pool = nn.AdaptiveAvgPool2d(1)
        self.light_encoder = LightokenEncoder(
            token_dim=self.config.light_dim,
            fourier_features=self.config.fourier_features,
            fourier_sigma=self.config.fourier_sigma,
            max_lights=self.config.max_lights,
        )
        fusion_in = int(self.config.visual_channels[-1]) + int(self.config.light_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
        )
        self.box_head = nn.Linear(self.config.hidden_dim, 4)
        self.presence_head = nn.Linear(self.config.hidden_dim, 1)

    def forward(
        self,
        source: torch.Tensor,
        object_mask: torch.Tensor,
        attrs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if source.ndim != 4 or source.shape[1] != 3:
            raise ValueError(f"source must be [B,3,H,W], got {tuple(source.shape)}")
        if object_mask.ndim == 3:
            object_mask = object_mask.unsqueeze(1)
        if object_mask.ndim != 4 or object_mask.shape[1] != 1:
            raise ValueError(f"object_mask must be [B,1,H,W], got {tuple(object_mask.shape)}")
        if source.shape[0] != object_mask.shape[0] or source.shape[-2:] != object_mask.shape[-2:]:
            raise ValueError("source and object_mask batch/spatial shapes must match")

        visual_pieces = [source, object_mask.to(dtype=source.dtype)]
        if self.config.use_coordconv:
            height, width = source.shape[-2:]
            y = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=source.dtype)
            x = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=source.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(source.shape[0], -1, -1, -1)
            visual_pieces.append(coordinates)
        visual = torch.cat(visual_pieces, dim=1)
        visual = self.visual_pool(self.visual_encoder(visual)).flatten(1)
        light_tokens = self.light_encoder(
            attrs,
            batch_size=int(source.shape[0]),
            device=source.device,
            dtype=source.dtype,
        )
        light = light_tokens.mean(dim=1)
        hidden = self.fusion(torch.cat([visual, light], dim=1))
        endpoints = self.box_head(hidden).sigmoid()
        x_pair = endpoints[:, (0, 2)]
        y_pair = endpoints[:, (1, 3)]
        x1, x2 = x_pair.min(dim=1).values, x_pair.max(dim=1).values
        y1, y2 = y_pair.min(dim=1).values, y_pair.max(dim=1).values
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        presence_logits = self.presence_head(hidden).squeeze(1)
        return {"boxes": boxes, "presence_logits": presence_logits}


def box_iou_xyxy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 4:
        raise ValueError(f"Expected matching [B,4] boxes, got {tuple(pred.shape)} and {tuple(target.shape)}")
    left_top = torch.maximum(pred[:, :2], target[:, :2])
    right_bottom = torch.minimum(pred[:, 2:], target[:, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=1)
    pred_area = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(dim=1)
    union = pred_area + target_area - intersection
    return intersection / union.clamp_min(float(eps))


def coshadow_box_loss(
    outputs: Mapping[str, torch.Tensor],
    target_boxes: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    l1_weight: float = 1.0,
    iou_weight: float = 1.0,
    presence_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    boxes = outputs["boxes"].float()
    presence_logits = outputs["presence_logits"].float()
    target_boxes = target_boxes.to(device=boxes.device, dtype=torch.float32)
    valid = target_valid.to(device=boxes.device, dtype=torch.bool).flatten()
    if boxes.shape != target_boxes.shape or valid.numel() != boxes.shape[0]:
        raise ValueError("Prediction and target batch shapes do not match")

    presence_raw = F.binary_cross_entropy_with_logits(presence_logits, valid.float())
    if bool(valid.any()):
        valid_pred = boxes[valid]
        valid_target = target_boxes[valid]
        l1_raw = F.l1_loss(valid_pred, valid_target)
        iou_values = box_iou_xyxy(valid_pred, valid_target)
        iou_raw = 1.0 - iou_values.mean()
        mean_iou = iou_values.mean()
    else:
        zero = boxes.sum() * 0.0
        l1_raw = zero
        iou_raw = zero
        mean_iou = zero.detach()
    total = (
        float(l1_weight) * l1_raw
        + float(iou_weight) * iou_raw
        + float(presence_weight) * presence_raw
    )
    with torch.no_grad():
        presence_accuracy = ((presence_logits >= 0) == valid).float().mean()
    metrics = {
        "loss": total.detach(),
        "l1": l1_raw.detach(),
        "iou_loss": iou_raw.detach(),
        "mean_iou": mean_iou.detach(),
        "presence_bce": presence_raw.detach(),
        "presence_accuracy": presence_accuracy.detach(),
        "valid_fraction": valid.float().mean().detach(),
    }
    return total, metrics


def quantize_boxes_xyxy(boxes: torch.Tensor, *, bins: int = 16) -> torch.Tensor:
    if int(bins) < 2:
        raise ValueError("bins must be at least 2")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be [B,4], got {tuple(boxes.shape)}")
    # Use round-half-up, matching the metadata builder.  torch.round uses
    # bankers rounding, which would disagree at exact half-bin boundaries.
    scaled = boxes.float().clamp(0, 1) * (int(bins) - 1)
    return torch.floor(scaled + 0.5).to(dtype=torch.long)


def _load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, Mapping):
        for key in ("state_dict", "model", "module"):
            nested = loaded.get(key)
            if isinstance(nested, Mapping):
                loaded = nested
                break
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Checkpoint must contain a state mapping: {path}")
    return {str(key).removeprefix("module."): value for key, value in loaded.items() if isinstance(value, torch.Tensor)}


def load_coshadow_box_predictor(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CoShadowBoxPredictor:
    path = Path(checkpoint)
    if path.is_dir():
        config_path = path / "config.json"
        candidates = [path / "model.safetensors", path / "model.pt"]
        model_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if model_path is None:
            raise FileNotFoundError(f"No model.safetensors or model.pt under {path}")
    else:
        model_path = path
        config_path = path.with_name("config.json")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    config_data: Mapping[str, Any] | None = None
    if config_path.is_file():
        loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
        config_data = loaded_config.get("model", loaded_config)
    model = CoShadowBoxPredictor(config_data)
    state = _load_tensor_state(model_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Box predictor checkpoint mismatch: missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.to(device=device, dtype=dtype)
    return model
