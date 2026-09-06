from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn


LIGHTOKEN_NAMES: tuple[str, ...] = ("a", "dg", "x", "y", "z", "r", "g", "b", "lambda", "d", "t")
GLOBAL_LIGHTOKEN_NAMES: tuple[str, ...] = ("a", "dg", "t")
PER_LIGHTOKEN_NAMES: tuple[str, ...] = ("x", "y", "z", "r", "g", "b", "lambda", "d")
ATTR_ALIASES = {
    "ambient": "a",
    "ambient_scale": "a",
    "ambient_scale_out": "a",
    "ambient_scale_delta": "a",
    "diffuse_gain": "dg",
    "spread_delta": "dg",
    "intensity": "lambda",
    "radius": "d",
    "spread": "d",
    "spread_out": "d",
    "transition": "t",
    "transition_on": "t",
    "red": "r",
    "green": "g",
    "blue": "b",
}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _triple(value: Any) -> tuple[float | None, float | None, float | None]:
    if value is None:
        return None, None, None
    values = list(value) if isinstance(value, Iterable) and not isinstance(value, (str, bytes)) else [value]
    values = (values + [None, None, None])[:3]
    return _finite_float(values[0]), _finite_float(values[1]), _finite_float(values[2])


def compact_direct_attrs(attrs: Mapping[str, Any] | None) -> dict[str, float]:
    if not attrs:
        return {}
    result: dict[str, float] = {}
    if "position" in attrs:
        for name, value in zip(("x", "y", "z"), _triple(attrs.get("position")), strict=True):
            if value is not None:
                result[name] = value
    if "color" in attrs:
        for name, value in zip(("r", "g", "b"), _triple(attrs.get("color")), strict=True):
            if value is not None:
                result[name] = value
    for key, value in attrs.items():
        name = ATTR_ALIASES.get(str(key), str(key))
        if name in LIGHTOKEN_NAMES and (number := _finite_float(value)) is not None:
            result[name] = number
    return result


def compact_attrs(attrs: Mapping[str, Any] | None) -> dict[str, float]:
    if not attrs:
        return {}
    result = compact_direct_attrs(attrs)
    if "lights" in attrs:
        lights = list(attrs.get("lights") or [])
        if lights and isinstance(lights[0], Mapping):
            result.update(compact_direct_attrs(lights[0]))
    return result


def parse_attrs_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        loaded = json.loads(text) if text else {}
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    return dict(value)


def attrs_from_batch(data: Mapping[str, Any], key: str = "attrs_json") -> list[dict[str, Any]]:
    value = data.get(key, data.get("attrs"))
    if value is None:
        return [{}]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        return [parse_attrs_json(value)]
    if isinstance(value, Sequence):
        return [parse_attrs_json(item) for item in value]
    return [parse_attrs_json(value)]


class GaussianFourierProjection(nn.Module):
    def __init__(self, features: int = 512, sigma: float = 5.0, num_attributes: int = len(LIGHTOKEN_NAMES)) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randn(int(num_attributes), int(features)) * float(sigma))

    @property
    def out_dim(self) -> int:
        return int(self.weight.shape[-1]) * 2

    def forward(self, values: torch.Tensor, attribute_ids: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1, 1)
        attribute_ids = attribute_ids.reshape(-1).to(device=values.device, dtype=torch.long)
        weight = self.weight.to(device=values.device, dtype=values.dtype)[attribute_ids]
        phases = 2.0 * math.pi * values * weight
        return torch.cat([phases.sin(), phases.cos()], dim=-1)


class LightokenEncoder(nn.Module):
    """TokenLight numeric light encoder with one projection per emitted token."""

    def __init__(
        self,
        token_dim: int,
        *,
        token_names: Sequence[str] = LIGHTOKEN_NAMES,
        fourier_features: int = 512,
        fourier_sigma: float = 5.0,
        max_lights: int = 1,
        null_value: float = -1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_token_names = tuple(token_names)
        self.max_lights = max(1, int(max_lights))
        if self.max_lights == 1:
            self.global_token_names: tuple[str, ...] = ()
            self.light_token_names: tuple[str, ...] = ()
            self.token_names = self.base_token_names
        else:
            self.global_token_names = tuple(name for name in GLOBAL_LIGHTOKEN_NAMES if name in self.base_token_names)
            self.light_token_names = tuple(name for name in PER_LIGHTOKEN_NAMES if name in self.base_token_names)
            self.token_names = self.global_token_names + tuple(
                f"light{slot}_{name}"
                for slot in range(self.max_lights)
                for name in self.light_token_names
            )
        self.token_dim = int(token_dim)
        self.null_value = float(null_value)
        self.dropout = float(dropout)
        self.fourier = GaussianFourierProjection(
            features=fourier_features,
            sigma=fourier_sigma,
            num_attributes=len(self.token_names),
        )
        self.projections = nn.ModuleList(
            nn.Linear(self.fourier.out_dim, self.token_dim) for _ in self.token_names
        )

    def forward(
        self,
        attrs: torch.Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        drop_light: bool | torch.Tensor | Sequence[bool] = False,
    ) -> torch.Tensor:
        param = self.projections[0].weight
        device = torch.device(device) if device is not None else param.device
        dtype = dtype or param.dtype
        values = self._values(attrs, batch_size=batch_size, device=device)
        batch = values.shape[0]
        valid = torch.isfinite(values)
        drop_mask = self._drop_mask(drop_light, batch=batch, device=device)
        if self.training and self.dropout > 0:
            drop_mask = drop_mask | (torch.rand(batch, device=device) < self.dropout)
        valid = valid & ~drop_mask[:, None]

        safe_values = torch.nan_to_num(values, nan=0.0).to(dtype=dtype)
        attribute_ids = torch.arange(len(self.token_names), device=device).expand(batch, -1)
        encoded = self.fourier(safe_values.reshape(-1), attribute_ids.reshape(-1))
        encoded = encoded.reshape(batch, len(self.token_names), -1)
        tokens = torch.stack(
            [projection(encoded[:, index]) for index, projection in enumerate(self.projections)],
            dim=1,
        )
        return torch.where(valid[..., None], tokens, torch.full_like(tokens, self.null_value))

    def _values(
        self,
        attrs: torch.Tensor | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
        *,
        batch_size: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(attrs, torch.Tensor):
            values = attrs.to(device=device, dtype=torch.float32)
            values = values.unsqueeze(0) if values.ndim == 1 else values
            if batch_size is not None and values.shape[0] == 1 and batch_size > 1:
                values = values.expand(batch_size, -1).contiguous()
            if values.shape[-1] != len(self.token_names):
                raise ValueError(f"Expected {len(self.token_names)} light attr values, got {values.shape[-1]}")
            return values
        attrs_list = self._attrs_list(attrs, batch_size=batch_size)
        values = torch.full((len(attrs_list), len(self.token_names)), float("nan"), device=device)
        for row, item in enumerate(attrs_list):
            if self.max_lights == 1:
                compact = compact_attrs(item)
                for col, name in enumerate(self.token_names):
                    if (number := _finite_float(compact.get(name))) is not None:
                        values[row, col] = number
            else:
                offset = 0
                global_attrs = compact_direct_attrs(item)
                for name in self.global_token_names:
                    if (number := _finite_float(global_attrs.get(name))) is not None:
                        values[row, offset] = number
                    offset += 1
                lights = self._lights(item)
                for slot in range(self.max_lights):
                    light_attrs = lights[slot] if slot < len(lights) else {}
                    for name in self.light_token_names:
                        if (number := _finite_float(light_attrs.get(name))) is not None:
                            values[row, offset] = number
                        offset += 1
        return values

    def _attrs_list(
        self,
        attrs: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
        *,
        batch_size: int | None,
    ) -> list[dict[str, Any]]:
        if attrs is None:
            attrs_list: list[dict[str, Any]] = [{}]
        elif isinstance(attrs, Mapping) or isinstance(attrs, (str, bytes)):
            attrs_list = [parse_attrs_json(attrs)]
        else:
            attrs_list = [parse_attrs_json(item) for item in attrs]
        if batch_size is None or len(attrs_list) == batch_size:
            return attrs_list
        if len(attrs_list) == 1:
            return attrs_list * int(batch_size)
        raise ValueError(f"Got {len(attrs_list)} attr records for batch size {batch_size}")

    def _lights(self, attrs: Mapping[str, Any]) -> list[dict[str, float]]:
        if "lights" in attrs:
            lights = attrs.get("lights") or []
            if isinstance(lights, Sequence) and not isinstance(lights, (str, bytes)):
                return [
                    compact_direct_attrs(light)
                    for light in list(lights)[: self.max_lights]
                    if isinstance(light, Mapping)
                ]
        direct = compact_direct_attrs(attrs)
        if any(name in direct for name in self.light_token_names):
            return [direct]
        return []

    @staticmethod
    def _drop_mask(drop_light: bool | torch.Tensor | Sequence[bool], *, batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(drop_light, torch.Tensor):
            mask = drop_light.to(device=device, dtype=torch.bool).flatten()
            return mask.expand(batch) if mask.numel() == 1 else mask
        if isinstance(drop_light, Sequence) and not isinstance(drop_light, (str, bytes)):
            return torch.tensor(list(drop_light), device=device, dtype=torch.bool).flatten()
        return torch.full((batch,), bool(drop_light), device=device, dtype=torch.bool)
