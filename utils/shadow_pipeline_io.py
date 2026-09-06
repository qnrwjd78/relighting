"""Shared path and serialization helpers for the shadow-refinement pipeline.

The helpers deliberately keep renderer ground-truth masks separate from model
predictions.  A row prepared by this module can therefore be handed to Wan
with ``predicted_shadow_mask`` without silently falling back to ``shadow_mask``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


WORKSPACE = Path(__file__).resolve().parents[1]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            value.setdefault("_manifest_index", len(rows))
            rows.append(value)
    return rows


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Atomically replace *path* with compact JSONL and return its row count."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return count


def resolve_path(value: str | Path, *, base_path: str | Path | None = None) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    if base_path is not None:
        under_base = Path(base_path) / raw
        if under_base.exists():
            return under_base
    return WORKSPACE / raw


def scene_id(row: Mapping[str, Any]) -> str:
    value = row.get("scene_id") or row.get("scene_folder")
    if isinstance(value, str) and value:
        return value
    raise KeyError("Manifest row has no scene_id or scene_folder")


def sample_name(row: Mapping[str, Any]) -> str:
    value = row.get("sample_name")
    if isinstance(value, str) and value:
        return value
    light_id = row.get("light_id")
    if light_id is not None:
        return f"light_{int(light_id):03d}"
    return f"row_{int(row.get('_manifest_index', 0)):06d}"


def sample_key(row: Mapping[str, Any]) -> str:
    """Stable, filesystem-safe identity used by all shadow cache stages."""

    return f"{scene_id(row)}__{sample_name(row)}"


def baseline_prediction_name(row: Mapping[str, Any]) -> str:
    light_id = row.get("light_id")
    if light_id is None:
        return f"{scene_id(row)}.png"
    return f"{scene_id(row)}_light_{int(light_id):03d}.png"


def stable_scene_unit(scene: str, *, seed: int) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{scene}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def stable_row_rank(row: Mapping[str, Any], *, seed: int) -> bytes:
    identity = f"{int(seed)}:{scene_id(row)}:{sample_name(row)}"
    return hashlib.sha256(identity.encode("utf-8")).digest()


def light_position(row: Mapping[str, Any]) -> list[float]:
    value = row.get("light_position")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        position = [float(component) for component in value]
        if not all(math.isfinite(component) for component in position):
            raise ValueError(f"Non-finite target light position for {sample_key(row)}")
        return position
    attrs = row.get("attrs_json")
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
    if isinstance(attrs, Mapping):
        lights = attrs.get("lights")
        if isinstance(lights, list) and lights and isinstance(lights[0], Mapping):
            light = lights[0]
            position = [float(light[axis]) for axis in ("x", "y", "z")]
            if not all(math.isfinite(component) for component in position):
                raise ValueError(
                    f"Non-finite target light position for {sample_key(row)}"
                )
            return position
    raise KeyError(f"No usable target light position for {sample_key(row)}")


def cache_npz_path(cache_root: str | Path, row: Mapping[str, Any]) -> Path:
    return Path(cache_root) / scene_id(row) / f"{sample_name(row)}.npz"


def _safe_stem(value: str) -> str:
    stem = Path(Path(value).name).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not safe:
        raise ValueError(f"Cannot form a cache stem from {value!r}")
    return safe


def adaptershadow_cache_paths(
    cache_root: str | Path,
    row: Mapping[str, Any],
    *,
    source_path: str | Path,
) -> dict[str, Path]:
    """Return paths matching ``scripts/run_shadowadapter_cache.py`` exactly."""

    root = Path(cache_root)
    prediction_name = baseline_prediction_name(row)
    target_stem = _safe_stem(prediction_name)
    source = Path(source_path).resolve()
    source_digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:16]
    return {
        "adapter_target_cache": root / "targets" / f"{target_stem}.npz",
        "adapter_delta_cache": root / "deltas" / f"{target_stem}.npz",
        "adapter_source_cache": root
        / "sources"
        / f"{source_digest}_{_safe_stem(source.name)}.npz",
    }
