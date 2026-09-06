from __future__ import annotations

"""MoGe-3 point-map conditioning for TokenLight delta-flow training.

This file is deliberately self-contained at the integration boundary: it adds
no changes to the existing TokenLight Python modules.  It provides three
commands:

* ``precompute``: run frozen MoGe-3 and cache only camera-space point maps and
  validity masks, sharded safely across ``torchrun`` workers;
* ``verify-cache``: validate every unique source image in a metadata file;
* ``train``: train TokenLight with separate point-XYZ and point-to-light
  direction token streams plus the existing same-scene delta-flow objective
  (lambda 0.2).

Only point XYZ and point-to-light direction are presented to the DiT.  The
cached validity map is used solely to zero invalid pixels.  MoGe normal/depth
outputs are not used.  During training, a random rectangular region of the
source token grid is dropped while the point/direction streams remain intact.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MOGE_REPO_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"
MOGE_CHECKPOINT_ID = "Ruicheng/moge-3-vitl"
POINTMAP_SCHEMA = "tokenlight_moge3_pointmap_v1"
LOSS_IMPL_VERSION = "tokenlight_moge3_point_direction_streams_source_cutout_delta_flow_lambda0p2_v3"
DEFAULT_CONFIG = "configs/train_480/rgb_baseline_15ep_b8_ga40.json"
DEFAULT_METADATA = "data_train/objaverse_fixed32_480/metadata.jsonl"
DEFAULT_DATASET_ROOT = "data/objaverse_fixed32_png"
DEFAULT_CACHE_ROOT = "data/moge3_pointmap_fixed32_480"
DEFAULT_MOGE_REPO = "weights/MoGe"
DEFAULT_MOGE_WEIGHTS = "weights/moge-3-vitl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _resolve_source(source: str, dataset_root: Path) -> Path:
    path = Path(str(source))
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def _source_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def _unique_sources(metadata: Path, dataset_root: Path) -> list[tuple[str, Path]]:
    unique: dict[str, Path] = {}
    for row in _read_jsonl(metadata):
        source = str(row.get("input_image") or "").strip()
        if not source:
            raise KeyError(f"metadata row is missing input_image: {row.get('scene_id', '<unknown>')}")
        resolved = _resolve_source(source, dataset_root)
        unique[str(resolved)] = resolved
    return [(_source_key(path), path) for path in sorted(unique.values(), key=str)]


def _rank_world() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if rank < 0 or world < 1 or rank >= world:
        raise ValueError(f"invalid distributed environment: rank={rank}, world={world}")
    return rank, world, local_rank


def _install_triton32_flexgemm_compat() -> None:
    """Provide the dtype API used by current FlexGEMM on Triton 3.2.

    TokenLight's pinned PyTorch 2.6 environment brings Triton 3.2, while the
    MoGe-3 FlexGEMM commit also supports newer Triton versions whose dtype has
    an ``itemsize`` property.  Adding the equivalent read-only property avoids
    replacing TokenLight's PyTorch/Triton stack.
    """

    import triton.language as tl

    dtype_class = type(tl.int32)
    if not hasattr(dtype_class, "itemsize"):
        setattr(
            dtype_class,
            "itemsize",
            property(lambda value: int(value.primitive_bitwidth) // 8),
        )


def _activate_isolated_moge_triton() -> None:
    """Switch only the offline precompute process to Triton 3.3."""

    import importlib

    runtime = Path(
        os.environ.get(
            "TOKENLIGHT_MOGE_TRITON_RUNTIME",
            str(REPO_ROOT / "weights/moge3_triton_runtime"),
        )
    ).resolve()
    if not (runtime / "triton").is_dir():
        return
    for module_name in list(sys.modules):
        if module_name == "triton" or module_name.startswith("triton."):
            del sys.modules[module_name]
    sys.path.insert(0, str(runtime))
    importlib.invalidate_caches()


def _load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device=device)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _validate_cache_file(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"points_cv", "valid"}:
            raise ValueError(f"unexpected fields in {path}: {sorted(data.files)}")
        points = data["points_cv"]
        valid = data["valid"]
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"invalid point shape in {path}: {points.shape}")
    if valid.shape != points.shape[:2]:
        raise ValueError(f"point/mask shape mismatch in {path}: {points.shape}, {valid.shape}")
    if points.dtype != np.float32 or valid.dtype != np.uint8:
        raise ValueError(f"invalid dtypes in {path}: {points.dtype}, {valid.dtype}")
    valid_bool = valid.astype(bool)
    if not np.isfinite(points).all():
        raise ValueError(f"non-finite point in {path}")
    if valid_bool.any() and not (points[..., 2][valid_bool] > 0).all():
        raise ValueError(f"non-positive valid camera Z in {path}")
    return int(points.shape[0]), int(points.shape[1])


def precompute_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Precompute MoGe-3 point maps")
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--dataset-base-path", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--moge-repo", default=DEFAULT_MOGE_REPO)
    parser.add_argument("--moge-weights", default=DEFAULT_MOGE_WEIGHTS)
    parser.add_argument("--fov-x", type=float, default=39.6)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N unique sources (0 = all)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv))

    rank, world, local_rank = _rank_world()
    if not torch.cuda.is_available():
        raise RuntimeError("MoGe-3 precompute requires CUDA (FlexGEMM/Triton)")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    metadata = Path(args.metadata).resolve()
    dataset_root = Path(args.dataset_base_path).resolve()
    cache_root = Path(args.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    sources = _unique_sources(metadata, dataset_root)
    if int(args.limit) < 0:
        parser.error("--limit must be non-negative")
    if int(args.limit) > 0:
        sources = sources[: int(args.limit)]
    assigned = sources[rank::world]

    moge_repo = Path(args.moge_repo).resolve()
    weights = Path(args.moge_weights).resolve()
    if weights.is_dir():
        weights = weights / "model.pt"
    if not (moge_repo / "moge/model/v3.py").is_file():
        raise FileNotFoundError(f"MoGe-3 repository not found: {moge_repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"MoGe-3 checkpoint not found: {weights}")
    _activate_isolated_moge_triton()
    _install_triton32_flexgemm_compat()
    sys.path.insert(0, str(moge_repo))
    from moge.model.v3 import MoGeModel  # type: ignore  # noqa: E402

    print(f"[rank {rank}/{world}] loading MoGe-3 on {device}; sources={len(assigned)}/{len(sources)}")
    model = MoGeModel.from_pretrained(str(weights)).to(device).eval()
    records: list[dict[str, Any]] = []
    for item_index, (key, source_path) in enumerate(assigned, start=1):
        if not source_path.is_file():
            raise FileNotFoundError(f"source image not found: {source_path}")
        relative_cache = Path(key[:2]) / f"{key}.npz"
        cache_path = cache_root / relative_cache
        if cache_path.is_file() and not args.overwrite:
            height, width = _validate_cache_file(cache_path)
        else:
            image = _load_rgb_tensor(source_path, device)
            output = model.infer(
                image,
                fov_x=float(args.fov_x),
                force_projection=True,
                apply_mask=False,
                refine_steps=int(args.refine_steps),
                resolution_level=int(args.resolution_level),
                use_fp16=bool(args.fp16),
            )
            points = output["points"].detach().float()
            mask = output["mask"].detach().bool()
            valid = mask & torch.isfinite(points).all(dim=-1) & (points[..., 2] > 0)
            points = torch.where(valid[..., None], points, torch.zeros_like(points))
            points_np = points.cpu().numpy().astype(np.float32, copy=False)
            valid_np = valid.cpu().numpy().astype(np.uint8, copy=False)
            _atomic_npz(cache_path, points_cv=points_np, valid=valid_np)
            height, width = _validate_cache_file(cache_path)
        records.append(
            {
                "cache": str(relative_cache),
                "fov_x_degrees": float(args.fov_x),
                "height": height,
                "moge_checkpoint": MOGE_CHECKPOINT_ID,
                "moge_repo_commit": MOGE_REPO_COMMIT,
                "refine_steps": int(args.refine_steps),
                "schema": POINTMAP_SCHEMA,
                "source": str(source_path),
                "source_key": key,
                "width": width,
            }
        )
        print(f"[rank {rank}] {item_index}/{len(assigned)} {source_path} -> {relative_cache}")

    index_path = cache_root / f"index.rank{rank:05d}.jsonl"
    temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, index_path)
    print(f"[rank {rank}] complete: {len(records)} point maps, index={index_path}")


# Keep the MoGe offline worker independent of TokenLight/DeepSpeed imports so
# its isolated Triton runtime cannot affect (or be affected by) training.
if len(sys.argv) > 1 and sys.argv[1] == "precompute":
    precompute_main(sys.argv[2:])
    raise SystemExit(0)


from model import wan as legacy  # noqa: E402
from model import train as base  # noqa: E402
from model import train_decoder_safe as safe  # noqa: E402
from model import train_delta_flow as delta  # noqa: E402
from model.light_encoder import LightokenEncoder, parse_attrs_json  # noqa: E402
from model.wan import TokenLightTypeEmbedding  # noqa: E402
from model.wan_spatial import _promote_clean_prefix_t_mod  # noqa: E402


class PointMapCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.index: dict[str, Path] = {}
        index_files = sorted(self.root.glob("index.rank*.jsonl"))
        if not index_files:
            raise FileNotFoundError(f"no MoGe point-map indices found in {self.root}")
        for index_path in index_files:
            for row in _read_jsonl(index_path):
                if row.get("schema") != POINTMAP_SCHEMA:
                    raise ValueError(f"unsupported point-map schema in {index_path}: {row.get('schema')}")
                source = str(Path(str(row["source"])).resolve())
                cache_path = self.root / str(row["cache"])
                previous = self.index.get(source)
                if previous is not None and previous != cache_path:
                    raise ValueError(f"duplicate cache entries for {source}: {previous}, {cache_path}")
                self.index[source] = cache_path

    def load(self, source: Path) -> tuple[torch.Tensor, torch.Tensor]:
        resolved = str(source.resolve())
        path = self.index.get(resolved)
        if path is None:
            raise KeyError(f"missing MoGe point map for {resolved}")
        with np.load(path, allow_pickle=False) as data:
            points = np.array(data["points_cv"], dtype=np.float32, copy=True)
            valid = np.array(data["valid"], dtype=np.uint8, copy=True)
        if points.ndim != 3 or points.shape[-1] != 3 or valid.shape != points.shape[:2]:
            raise ValueError(f"malformed point-map cache: {path}")
        return torch.from_numpy(points).permute(2, 0, 1), torch.from_numpy(valid.astype(bool))


class PointMapCachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, *, cache_root: str | Path, dataset_root: str | Path) -> None:
        self.dataset = dataset
        self.cache = PointMapCache(cache_root)
        self.dataset_root = Path(dataset_root).resolve()
        self.data = dataset.data
        self.repeat = getattr(dataset, "repeat", 1)
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.is_vae_latent_cache_dataset = getattr(dataset, "is_vae_latent_cache_dataset", False)

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_item(self, index: int) -> dict[str, Any]:
        item = self.dataset[int(index)]
        source = _resolve_source(str(item["input_image"]), self.dataset_root)
        object_mask_path = _resolve_source(str(item["mask"]), self.dataset_root)
        if not object_mask_path.is_file():
            raise FileNotFoundError(f"object mask not found: {object_mask_path}")
        points, valid = self.cache.load(source)
        object_mask = np.asarray(Image.open(object_mask_path).convert("L"), dtype=np.uint8) >= 128
        if object_mask.shape != tuple(valid.shape):
            raise ValueError(
                f"object mask/point-map shape mismatch for {source}: {object_mask.shape} != {tuple(valid.shape)}"
            )
        item["_tokenlight_moge_points"] = points
        item["_tokenlight_moge_valid"] = valid
        item["_tokenlight_moge_object_mask"] = torch.from_numpy(object_mask.copy())
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._load_item(index)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self._load_item(int(index)) for index in indices]


def verify_cache_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Verify MoGe-3 point-map cache coverage")
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--dataset-base-path", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="Verify at most N sources (0 = all)")
    args = parser.parse_args(list(argv))
    sources = _unique_sources(Path(args.metadata).resolve(), Path(args.dataset_base_path).resolve())
    if int(args.limit) < 0:
        parser.error("--limit must be non-negative")
    if int(args.limit) > 0:
        sources = sources[: int(args.limit)]
    cache = PointMapCache(args.cache_root)
    missing: list[str] = []
    for _, path in sources:
        try:
            points, valid = cache.load(path)
            if points.shape[1:] != valid.shape:
                raise ValueError(f"shape mismatch for {path}")
        except (KeyError, OSError, ValueError) as exc:
            missing.append(f"{path}: {exc}")
    if missing:
        raise RuntimeError(f"point-map cache verification failed ({len(missing)}/{len(sources)}):\n" + "\n".join(missing[:20]))
    print(f"point-map cache verified: {len(sources)} unique source images")


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DenseMapTokenEncoder(nn.Module):
    def __init__(self, input_channels: int, token_dim: int, hidden_channels: int = 64) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        middle = hidden_channels * 2
        self.input_channels = int(input_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_channels, 7, stride=4, padding=3),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, middle, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(middle), middle),
            nn.SiLU(),
            nn.Conv2d(middle, middle, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(middle), middle),
            nn.SiLU(),
        )
        self.projection = nn.Conv2d(middle, int(token_dim), 1)
        self.norm = nn.LayerNorm(int(token_dim), elementwise_affine=False)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, value: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.input_channels:
            raise ValueError(f"expected [B,{self.input_channels},H,W], got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError("point-map feature contains NaN or Inf")
        parameter = next(self.parameters())
        features = self.stem(value.to(dtype=parameter.dtype))
        features = F.adaptive_avg_pool2d(features, output_size)
        tokens = self.projection(features).flatten(2).transpose(1, 2).contiguous()
        return self.norm(tokens)


MOGE_TYPE_POINT = 0
MOGE_TYPE_DIRECTION = 1


class MoGeStreamTypeEmbedding(nn.Module):
    """Type embeddings for the two clean MoGe spatial condition streams."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, int(token_dim))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        value = self.embedding.weight[int(type_id)].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + value.view(1, 1, -1)


def _source_token_object_dropout(
    tokens: torch.Tensor,
    grid: tuple[int, int, int],
    object_mask: torch.Tensor,
    *,
    probability: float,
    region_fraction: float,
    pair_indices: torch.Tensor | None,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop equal fractions inside/outside the object, shared within delta pairs."""

    batch, length, _ = tokens.shape
    frames, height, width = (int(value) for value in grid)
    if length != frames * height * width:
        raise ValueError(f"source token/grid mismatch: {length} != {frames}*{height}*{width}")
    if object_mask.ndim == 4 and object_mask.shape[1] == 1:
        object_mask = object_mask[:, 0]
    if object_mask.ndim != 3 or object_mask.shape[0] != batch:
        raise ValueError(f"object mask must be [B,H,W], got {tuple(object_mask.shape)}")
    object_grid = F.interpolate(
        object_mask[:, None].to(device=tokens.device, dtype=torch.float32),
        size=(height, width),
        mode="area",
    )[:, 0] >= 0.5
    dropped = torch.zeros((batch, frames, height, width), dtype=torch.bool, device=tokens.device)
    if not training or probability <= 0:
        zero = dropped.float().mean()
        return tokens, zero, zero, zero

    for index in range(batch):
        if float(torch.rand((), device=tokens.device)) >= probability:
            continue
        spatial_drop = torch.zeros((height, width), dtype=torch.bool, device=tokens.device)
        for region in (object_grid[index], ~object_grid[index]):
            candidates = region.flatten().nonzero(as_tuple=False).flatten()
            count = min(candidates.numel(), int(round(region_fraction * candidates.numel())))
            if count > 0:
                selected = candidates[torch.randperm(candidates.numel(), device=tokens.device)[:count]]
                spatial_drop.flatten()[selected] = True
        dropped[index] = spatial_drop[None].expand(frames, -1, -1)

    if pair_indices is not None:
        pairs = pair_indices.to(device=tokens.device, dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"delta pair indices must be [P,2], got {tuple(pairs.shape)}")
        if pairs.numel() and (int(pairs.min()) < 0 or int(pairs.max()) >= batch):
            raise IndexError("delta pair index is outside the source-token batch")
        for left, right in pairs.tolist():
            dropped[right] = dropped[left]

    flat = dropped.reshape(batch, length, 1)
    spatial_dropped = dropped.any(dim=1)
    object_count = object_grid.float().sum().clamp_min(1.0)
    background_count = (~object_grid).float().sum().clamp_min(1.0)
    object_fraction = (spatial_dropped & object_grid).float().sum() / object_count
    background_fraction = (spatial_dropped & ~object_grid).float().sum() / background_count
    return tokens.masked_fill(flat, 0), dropped.float().mean(), object_fraction, background_fraction


def _batch_bool(value: Any, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.bool).reshape(-1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = torch.tensor(list(value), device=device, dtype=torch.bool).reshape(-1)
    else:
        result = torch.full((batch,), bool(value), device=device, dtype=torch.bool)
    if result.numel() == 1 and batch != 1:
        result = result.expand(batch)
    if result.numel() != batch:
        raise ValueError(f"drop-light batch mismatch: {result.numel()} != {batch}")
    return result


def _light_positions(attrs: Any, *, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(attrs, Mapping) or isinstance(attrs, str):
        attrs = [attrs]
    if not isinstance(attrs, Sequence) or isinstance(attrs, (str, bytes)) or len(attrs) != batch:
        raise ValueError(f"expected {batch} TokenLight attribute records")
    parsed = [parse_attrs_json(item) for item in attrs]
    max_lights = max(1, max(len(item.get("lights", [])) for item in parsed))
    positions = torch.zeros(batch, max_lights, 3, dtype=torch.float32, device=device)
    slot_valid = torch.zeros(batch, max_lights, dtype=torch.bool, device=device)
    for batch_index, item in enumerate(parsed):
        lights = item.get("lights", [])
        for light_index, light in enumerate(lights):
            if not isinstance(light, Mapping):
                continue
            xyz = [float(light[name]) for name in ("x", "y", "z")]
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError(f"non-finite light position at batch {batch_index}, slot {light_index}")
            positions[batch_index, light_index] = torch.tensor(xyz, device=device)
            slot_valid[batch_index, light_index] = True
    return positions, slot_valid


class PointMapConditioner(nn.Module):
    def __init__(
        self,
        token_dim: int,
        *,
        hidden_channels: int = 64,
        camera_distance: float = 3.5,
        camera_metric_scale: float = 0.75,
        light_metric_scale: float = 14.0 / 15.0,
        initial_scale: float = 1.0,
        trainable_scale: bool = True,
    ) -> None:
        super().__init__()
        if camera_distance <= 0 or camera_metric_scale <= 0 or light_metric_scale <= 0 or initial_scale <= 0:
            raise ValueError("camera distance/scales and initial scale must be positive")
        self.camera_distance = float(camera_distance)
        self.camera_metric_scale = float(camera_metric_scale)
        self.light_metric_scale = float(light_metric_scale)
        self.point_encoder = DenseMapTokenEncoder(3, token_dim, hidden_channels)
        self.direction_encoder = DenseMapTokenEncoder(3, token_dim, hidden_channels)
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(float(initial_scale))), requires_grad=bool(trainable_scale)
        )

    def forward(
        self,
        points_cv: torch.Tensor,
        valid: torch.Tensor,
        attrs: Any,
        drop_light: Any,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if points_cv.ndim != 4 or points_cv.shape[1] != 3:
            raise ValueError(f"expected point map [B,3,H,W], got {tuple(points_cv.shape)}")
        if valid.ndim == 4 and valid.shape[1] == 1:
            valid = valid[:, 0]
        if valid.ndim != 3 or valid.shape != points_cv.shape[:1] + points_cv.shape[2:]:
            raise ValueError(f"point validity shape mismatch: {tuple(valid.shape)}")
        compute_dtype = self.point_encoder.projection.weight.dtype
        points = points_cv.to(dtype=compute_dtype)
        valid = valid.to(device=points.device, dtype=torch.bool)
        valid = valid & torch.isfinite(points).all(dim=1) & (points[:, 2] > 0)
        points = torch.where(valid[:, None], points, torch.zeros_like(points))
        alpha = self.log_scale.float().clamp(-5.0, 5.0).exp().to(dtype=compute_dtype)
        scaled = alpha * points
        # MoGe points are metric camera-space points.  fixed32 does not apply
        # one shared scale to the camera and lights: the canonical camera rig
        # is rendered at 0.75x, while point-light positions are rendered at
        # 14/15x (also recorded by world_radius / canonical_radius).  Keep the
        # dense branch in metric camera space instead of subtracting canonical
        # light coordinates directly from metric MoGe points.
        camera_distance_can = torch.as_tensor(self.camera_distance, device=points.device, dtype=compute_dtype)
        camera_scale = torch.as_tensor(self.camera_metric_scale, device=points.device, dtype=compute_dtype)
        light_scale = torch.as_tensor(self.light_metric_scale, device=points.device, dtype=compute_dtype)
        d0 = camera_distance_can * camera_scale
        point_map = torch.where(valid[:, None], scaled / d0, torch.zeros_like(scaled))
        point_tokens = self.point_encoder(point_map, output_size)

        positions_can, slot_valid = _light_positions(attrs, batch=points.shape[0], device=points.device)
        positions_can = positions_can.to(dtype=compute_dtype)
        # A_cv_to_can maps (X,Y,Z)_cv -> (X,Z,-Y)_can.  With separate
        # renderer scales, the exact metric camera-frame position is
        # A^T(s_light * L_can - s_camera * O_can), O_can=(0,-d,0).
        light_cv = torch.stack(
            (
                light_scale * positions_can[..., 0],
                -light_scale * positions_can[..., 2],
                light_scale * positions_can[..., 1] + d0,
            ),
            dim=-1,
        )
        drop = _batch_bool(drop_light, points.shape[0], points.device)
        direction_sum = torch.zeros_like(point_tokens)
        active_slots = torch.zeros(points.shape[0], dtype=compute_dtype, device=points.device)
        for light_index in range(light_cv.shape[1]):
            active = valid & slot_valid[:, light_index, None, None] & ~drop[:, None, None]
            vector = light_cv[:, light_index, :, None, None] - scaled
            distance = torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(1e-6)
            direction = vector / distance
            direction_map = torch.where(active[:, None], direction, torch.zeros_like(direction))
            encoded = self.direction_encoder(direction_map, output_size)
            slot_active = (slot_valid[:, light_index] & ~drop).to(compute_dtype)
            direction_sum = direction_sum + encoded * slot_active[:, None, None]
            active_slots = active_slots + slot_active
        direction_tokens = direction_sum / active_slots.clamp_min(1.0)[:, None, None]
        metrics = {
            "moge/alpha": alpha.detach(),
            "moge/point_token_rms": point_tokens.float().square().mean().sqrt().detach(),
            "moge/direction_token_rms": direction_tokens.float().square().mean().sqrt().detach(),
            "moge/valid_fraction": valid.float().mean().detach(),
        }
        return point_tokens, direction_tokens, metrics


def model_fn_wan_video_tokenlight_moge3(
    *,
    dit: nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    motion_controller: nn.Module | None = None,
    motion_bucket_id: torch.Tensor | None = None,
    tokenlight_light_encoder: LightokenEncoder | None = None,
    tokenlight_type_embedding: TokenLightTypeEmbedding | None = None,
    tokenlight_moge_conditioner: PointMapConditioner | None = None,
    tokenlight_moge_type_embedding: MoGeStreamTypeEmbedding | None = None,
    tokenlight_attrs: Any = None,
    tokenlight_drop_light: Any = False,
    tokenlight_source_latents: torch.Tensor | None = None,
    tokenlight_mask_latents: torch.Tensor | None = None,
    tokenlight_moge_points: torch.Tensor | None = None,
    tokenlight_moge_valid: torch.Tensor | None = None,
    tokenlight_moge_object_mask: torch.Tensor | None = None,
    tokenlight_delta_pair_indices: torch.Tensor | None = None,
    tokenlight_moge_source_dropout_prob: float = 0.3,
    tokenlight_moge_source_dropout_region_fraction: float = 0.3,
    tokenlight_moge_training: bool | None = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
) -> torch.Tensor:
    del kwargs
    from einops import rearrange
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    if getattr(dit, "seperated_timestep", False) and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep,
            ]
        ).flatten()
        t_head = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t_head).unflatten(2, (6, dit.dim))
    else:
        t_head = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t_head).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)
    batch = int(context.shape[0])
    x = latents if latents.shape[0] == batch else torch.cat([latents] * batch, dim=0)
    if y is not None and getattr(dit, "require_vae_embedding", True):
        x = torch.cat([x, legacy._repeat_to_batch(y, batch)], dim=1)
    if clip_feature is not None and getattr(dit, "require_clip_embedding", True):
        context = torch.cat([dit.img_emb(legacy._repeat_to_batch(clip_feature, batch)), context], dim=1)

    patches = dit.patchify(x, control_camera_latents_input)
    target_grid = patches.shape[2:]
    target_tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    target_tokens = legacy._add_type_embedding(target_tokens, tokenlight_type_embedding, legacy.TOKENLIGHT_TYPE_TARGET)
    target_freqs = legacy._freqs_for_grid(dit, target_grid, target_tokens.device)
    prefix_tokens: list[torch.Tensor] = []
    prefix_freqs: list[torch.Tensor] = []

    if tokenlight_source_latents is None:
        raise ValueError("MoGe point-map streams require tokenlight_source_latents")
    source_tokens, source_grid = legacy._patch_to_tokens(dit, tokenlight_source_latents, batch)
    # DiffSynth freezes the pipeline by calling ``pipe.eval()`` before it
    # attaches LoRA modules.  Consequently ``dit.training`` remains False
    # during LoRA training and must not be used to gate training-only source
    # dropout.  The owning training module passes its actual mode explicitly;
    # direct inference callers retain the safe ``dit.training`` fallback.
    source_dropout_training = (
        bool(dit.training)
        if tokenlight_moge_training is None
        else bool(tokenlight_moge_training)
    )
    if source_dropout_training:
        if tokenlight_moge_object_mask is None:
            raise ValueError("object-aware source dropout requires tokenlight_moge_object_mask during training")
        (
            source_tokens,
            source_dropout_fraction,
            source_object_dropout_fraction,
            source_background_dropout_fraction,
        ) = _source_token_object_dropout(
            source_tokens,
            source_grid,
            tokenlight_moge_object_mask,
            probability=float(tokenlight_moge_source_dropout_prob),
            region_fraction=float(tokenlight_moge_source_dropout_region_fraction),
            pair_indices=tokenlight_delta_pair_indices,
            training=True,
        )
    else:
        source_dropout_fraction = source_tokens.new_zeros(())
        source_object_dropout_fraction = source_tokens.new_zeros(())
        source_background_dropout_fraction = source_tokens.new_zeros(())
    source_tokens = legacy._add_type_embedding(source_tokens, tokenlight_type_embedding, legacy.TOKENLIGHT_TYPE_SOURCE)
    prefix_tokens.append(source_tokens)
    prefix_freqs.append(legacy._freqs_for_grid(dit, source_grid, target_tokens.device))

    if (
        tokenlight_moge_conditioner is None
        or tokenlight_moge_type_embedding is None
        or tokenlight_moge_points is None
        or tokenlight_moge_valid is None
    ):
        raise ValueError("MoGe point/direction stream inputs or modules are missing")
    point_tokens, direction_tokens, geometry_metrics = tokenlight_moge_conditioner(
        tokenlight_moge_points,
        tokenlight_moge_valid,
        tokenlight_attrs,
        tokenlight_drop_light,
        (int(source_grid[1]), int(source_grid[2])),
    )
    point_tokens = point_tokens.to(device=source_tokens.device, dtype=source_tokens.dtype)
    direction_tokens = direction_tokens.to(device=source_tokens.device, dtype=source_tokens.dtype)
    if point_tokens.shape != source_tokens.shape or direction_tokens.shape != source_tokens.shape:
        raise ValueError(
            "MoGe/source token mismatch: "
            f"point={point_tokens.shape}, direction={direction_tokens.shape}, source={source_tokens.shape}"
        )
    point_tokens = tokenlight_moge_type_embedding(point_tokens, MOGE_TYPE_POINT)
    direction_tokens = tokenlight_moge_type_embedding(direction_tokens, MOGE_TYPE_DIRECTION)
    spatial_freqs = legacy._freqs_for_grid(dit, source_grid, target_tokens.device)
    prefix_tokens.extend((point_tokens, direction_tokens))
    prefix_freqs.extend((spatial_freqs, spatial_freqs))
    tokenlight_moge_conditioner._last_metrics = {
        **geometry_metrics,
        "moge/source_dropout_fraction": source_dropout_fraction.detach(),
        "moge/source_object_dropout_fraction": source_object_dropout_fraction.detach(),
        "moge/source_background_dropout_fraction": source_background_dropout_fraction.detach(),
    }

    if tokenlight_mask_latents is not None:
        mask_tokens, mask_grid = legacy._patch_to_tokens(dit, tokenlight_mask_latents, batch)
        mask_tokens = legacy._add_type_embedding(mask_tokens, tokenlight_type_embedding, legacy.TOKENLIGHT_TYPE_MASK)
        prefix_tokens.append(mask_tokens)
        prefix_freqs.append(legacy._freqs_for_grid(dit, mask_grid, target_tokens.device))
    if tokenlight_light_encoder is not None:
        light_tokens = tokenlight_light_encoder(
            tokenlight_attrs,
            batch_size=batch,
            device=target_tokens.device,
            dtype=target_tokens.dtype,
            drop_light=tokenlight_drop_light,
        )
        light_tokens = legacy._add_type_embedding(light_tokens, tokenlight_type_embedding, legacy.TOKENLIGHT_TYPE_LIGHT)
        prefix_tokens.append(light_tokens)
        prefix_freqs.append(torch.ones(light_tokens.shape[1], 1, target_freqs.shape[-1], device=target_tokens.device, dtype=target_freqs.dtype))

    prefix_len = sum(tokens.shape[1] for tokens in prefix_tokens)
    x = torch.cat([*prefix_tokens, target_tokens], dim=1)
    freqs = torch.cat([*prefix_freqs, target_freqs], dim=0)
    t_mod = _promote_clean_prefix_t_mod(
        dit, t_mod, prefix_len=prefix_len, target_len=int(target_tokens.shape[1]), batch=batch
    )
    for block in dit.blocks:
        x = legacy.gradient_checkpoint_forward_compatible(
            block, use_gradient_checkpointing, use_gradient_checkpointing_offload, x, context, t_mod, freqs
        )
    x = dit.head(x[:, prefix_len:], t_head)
    return dit.unpatchify(x, target_grid)


def _stack_pointmaps(
    value: Any,
    *,
    batch: int,
    item_ndim: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim == item_ndim:
            tensor = value.unsqueeze(0)
        elif value.ndim == item_ndim + 1 and value.shape[0] == batch:
            tensor = value
        else:
            raise ValueError(f"malformed batched {name}: {tuple(value.shape)}")
    elif isinstance(value, list) and len(value) == batch and all(isinstance(item, torch.Tensor) for item in value):
        tensor = torch.stack(value, dim=0)
    else:
        raise TypeError(f"missing or malformed batched {name}")
    return tensor.to(device=device, non_blocking=bool(tensor.is_pinned()))


class TokenLightMoGe3TrainingModule(delta.TokenLightDeltaFlowTrainingModule):
    def __init__(
        self,
        *args,
        tokenlight_moge_hidden_channels: int = 64,
        tokenlight_moge_camera_distance: float = 3.5,
        tokenlight_moge_camera_metric_scale: float = 0.75,
        tokenlight_moge_light_metric_scale: float = 14.0 / 15.0,
        tokenlight_moge_initial_scale: float = 1.0,
        tokenlight_moge_trainable_scale: bool = True,
        tokenlight_moge_source_dropout_prob: float = 0.3,
        tokenlight_moge_source_dropout_region_fraction: float = 0.3,
        **kwargs,
    ) -> None:
        checkpoint = kwargs.get("lora_checkpoint") or kwargs.get("resume_from_checkpoint")
        super().__init__(*args, **kwargs)
        token_dim = int(self.pipe.dit.dim)
        self.moge_conditioner = PointMapConditioner(
            token_dim,
            hidden_channels=tokenlight_moge_hidden_channels,
            camera_distance=tokenlight_moge_camera_distance,
            camera_metric_scale=tokenlight_moge_camera_metric_scale,
            light_metric_scale=tokenlight_moge_light_metric_scale,
            initial_scale=tokenlight_moge_initial_scale,
            trainable_scale=tokenlight_moge_trainable_scale,
        )
        self.moge_type_embedding = MoGeStreamTypeEmbedding(token_dim)
        self.tokenlight_moge_source_dropout_prob = float(tokenlight_moge_source_dropout_prob)
        self.tokenlight_moge_source_dropout_region_fraction = float(
            tokenlight_moge_source_dropout_region_fraction
        )
        aux_state = base._load_checkpoint_state_dict(checkpoint)
        base._load_module_state_from_checkpoint(self.moge_conditioner, aux_state, "moge_conditioner")
        base._load_module_state_from_checkpoint(self.moge_type_embedding, aux_state, "moge_type_embedding")
        self.pipe.model_fn = self._tokenlight_model_fn

    def _tokenlight_model_fn(self, **kwargs):
        return model_fn_wan_video_tokenlight_moge3(
            **kwargs,
            tokenlight_light_encoder=self.light_encoder,
            tokenlight_type_embedding=self.tokenlight_type_embedding,
            tokenlight_moge_conditioner=self.moge_conditioner,
            tokenlight_moge_type_embedding=self.moge_type_embedding,
            tokenlight_moge_source_dropout_prob=self.tokenlight_moge_source_dropout_prob,
            tokenlight_moge_source_dropout_region_fraction=self.tokenlight_moge_source_dropout_region_fraction,
            tokenlight_moge_training=self.training,
        )

    def _attach_delta_inputs(self, inputs, data):
        inputs_shared, inputs_posi, inputs_nega = super()._attach_delta_inputs(inputs, data)
        batch = int(inputs_shared["input_latents"].shape[0])
        device = inputs_shared["input_latents"].device
        points = _stack_pointmaps(
            data.get("_tokenlight_moge_points"),
            batch=batch,
            item_ndim=3,
            name="MoGe points",
            device=device,
        )
        valid = _stack_pointmaps(
            data.get("_tokenlight_moge_valid"),
            batch=batch,
            item_ndim=2,
            name="MoGe valid",
            device=device,
        )
        object_mask = _stack_pointmaps(
            data.get("_tokenlight_moge_object_mask"),
            batch=batch,
            item_ndim=2,
            name="object mask",
            device=device,
        )
        if points.ndim != 4 or points.shape[1] != 3:
            raise ValueError(f"expected cached points [B,3,H,W], got {tuple(points.shape)}")
        if valid.ndim != 3:
            raise ValueError(f"expected cached valid [B,H,W], got {tuple(valid.shape)}")
        if object_mask.ndim != 3 or object_mask.shape != valid.shape:
            raise ValueError(f"expected object mask {tuple(valid.shape)}, got {tuple(object_mask.shape)}")
        inputs_shared["tokenlight_moge_points"] = points.float()
        inputs_shared["tokenlight_moge_valid"] = valid.bool()
        inputs_shared["tokenlight_moge_object_mask"] = object_mask.bool()
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        loss = super().forward(data, inputs=inputs)
        metrics = getattr(self.moge_conditioner, "_last_metrics", None)
        if isinstance(metrics, Mapping):
            current = getattr(self.pipe, "_tokenlight_loss_metrics", {})
            self.pipe._tokenlight_loss_metrics = {**current, **metrics}
        return loss

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        exported = super().export_trainable_state_dict(state_dict, remove_prefix=remove_prefix)
        for prefix in ("moge_conditioner.", "moge_type_embedding."):
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    exported[key] = value
        return exported


def training_parser(train_mode: str) -> argparse.ArgumentParser:
    parser = delta.delta_parser(train_mode)
    parser.add_argument("--tokenlight_moge_cache_root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--tokenlight_moge_hidden_channels", type=int, default=64)
    parser.add_argument("--tokenlight_moge_camera_distance", type=float, default=3.5)
    parser.add_argument("--tokenlight_moge_camera_metric_scale", type=float, default=0.75)
    parser.add_argument("--tokenlight_moge_light_metric_scale", type=float, default=14.0 / 15.0)
    parser.add_argument("--tokenlight_moge_initial_scale", type=float, default=1.0)
    parser.add_argument("--tokenlight_moge_trainable_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenlight_moge_source_dropout_prob", type=float, default=0.3)
    parser.add_argument("--tokenlight_moge_source_dropout_region_fraction", type=float, default=0.3)
    return parser


def parse_training_args(argv: Sequence[str]):
    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--train_mode", choices=("single", "zero3"), default="single")
        pre.add_argument("--config", default=None)
        pre_args, _ = pre.parse_known_args()
        config_path = pre_args.config or os.environ.get("TOKENLIGHT_TRAIN_CONFIG", DEFAULT_CONFIG)
        raw_config = base._load_json_config(config_path)
        parser = training_parser(pre_args.train_mode)
        base._apply_config_defaults(parser, raw_config)
        parser.set_defaults(
            config=config_path,
            train_mode=pre_args.train_mode,
            tokenlight_delta_loss_weight=0.2,
            gradient_accumulation_steps=1,
            output_path="outputs/train/exp_0/rgb_moge3_point_direction_streams_delta_flow_lambda0p2_15ep_b8_ga1_g5_gb40",
        )
        args = parser.parse_args()
    finally:
        sys.argv = original_argv
    raw_config = base._load_json_config(args.config)
    delta._validate_delta_args(args, raw_config, parser)
    if args.tokenlight_moge_hidden_channels < 8:
        parser.error("tokenlight_moge_hidden_channels must be >= 8")
    if (
        args.tokenlight_moge_camera_distance <= 0
        or args.tokenlight_moge_camera_metric_scale <= 0
        or args.tokenlight_moge_light_metric_scale <= 0
        or args.tokenlight_moge_initial_scale <= 0
    ):
        parser.error("MoGe camera distance/scales and initial scale must be positive")
    if not 0.0 <= args.tokenlight_moge_source_dropout_prob <= 1.0:
        parser.error("tokenlight_moge_source_dropout_prob must be in [0,1]")
    if not 0.0 < args.tokenlight_moge_source_dropout_region_fraction < 1.0:
        parser.error("tokenlight_moge_source_dropout_region_fraction must be in (0,1)")
    base._resolve_weight_paths(args)
    base._append_timestamp_to_output_path(args)
    return args, raw_config, raw_config


def _build_model(args, accelerator):
    return TokenLightMoGe3TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=getattr(args, "use_gradient_checkpointing_offload", False),
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu or getattr(args, "enable_model_cpu_offload", False) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tokenlight_light_tokens=args.tokenlight_light_tokens,
        tokenlight_attrs_key=args.tokenlight_attrs_key,
        tokenlight_token_dim=args.tokenlight_token_dim,
        tokenlight_fourier_features=args.tokenlight_fourier_features,
        tokenlight_fourier_sigma=args.tokenlight_fourier_sigma,
        tokenlight_max_lights=args.tokenlight_max_lights,
        tokenlight_light_dropout=args.tokenlight_light_dropout,
        tokenlight_cfg_drop_prob=args.tokenlight_cfg_drop_prob,
        tokenlight_source_tokens=args.tokenlight_source_tokens,
        tokenlight_mask_tokens=args.tokenlight_mask_tokens,
        prompt_context_cache_size=args.prompt_context_cache_size,
        tokenlight_rgb_latent_loss_weight=args.tokenlight_rgb_latent_loss_weight,
        tokenlight_rgb_decoder_loss_weight=args.tokenlight_rgb_decoder_loss_weight,
        tokenlight_rgb_decoder_transform=args.tokenlight_rgb_decoder_transform,
        tokenlight_decoder_loss_type=args.tokenlight_decoder_loss_type,
        tokenlight_decoder_luminance_eps=args.tokenlight_decoder_luminance_eps,
        tokenlight_decoder_charbonnier_eps=args.tokenlight_decoder_charbonnier_eps,
        tokenlight_rgb_decoder_full_loss_weight=args.tokenlight_rgb_decoder_full_loss_weight,
        tokenlight_rgb_decoder_shadow_loss_weight=args.tokenlight_rgb_decoder_shadow_loss_weight,
        tokenlight_rgb_decoder_direct_loss_weight=args.tokenlight_rgb_decoder_direct_loss_weight,
        tokenlight_rgb_decoder_shadow_mask_key=args.tokenlight_rgb_decoder_shadow_mask_key,
        tokenlight_rgb_decoder_direct_mask_key=args.tokenlight_rgb_decoder_direct_mask_key,
        tokenlight_decoder_min_region_fraction=args.tokenlight_decoder_min_region_fraction,
        tokenlight_decoder_max_samples_per_batch=args.tokenlight_decoder_max_samples_per_batch,
        tokenlight_decoder_gradient_checkpointing=args.tokenlight_decoder_gradient_checkpointing,
        tokenlight_decoder_warmup_steps=args.tokenlight_decoder_warmup_steps,
        tokenlight_decoder_ramp_steps=args.tokenlight_decoder_ramp_steps,
        tokenlight_mask_image_key=args.tokenlight_mask_image_key,
        tokenlight_extra_mask_image_keys=args.tokenlight_extra_mask_image_keys,
        tokenlight_extra_mask_latent_cache_dirs=args.tokenlight_extra_mask_latent_cache_dirs,
        tokenlight_delta_pairs_per_batch=args.tokenlight_delta_pairs_per_batch,
        tokenlight_delta_loss_weight=args.tokenlight_delta_loss_weight,
        tokenlight_delta_loss_type=args.tokenlight_delta_loss_type,
        tokenlight_delta_charbonnier_eps=args.tokenlight_delta_charbonnier_eps,
        tokenlight_delta_warmup_steps=args.tokenlight_delta_warmup_steps,
        tokenlight_delta_ramp_steps=args.tokenlight_delta_ramp_steps,
        tokenlight_delta_sigma_gamma=args.tokenlight_delta_sigma_gamma,
        tokenlight_moge_hidden_channels=args.tokenlight_moge_hidden_channels,
        tokenlight_moge_camera_distance=args.tokenlight_moge_camera_distance,
        tokenlight_moge_camera_metric_scale=args.tokenlight_moge_camera_metric_scale,
        tokenlight_moge_light_metric_scale=args.tokenlight_moge_light_metric_scale,
        tokenlight_moge_initial_scale=args.tokenlight_moge_initial_scale,
        tokenlight_moge_trainable_scale=args.tokenlight_moge_trainable_scale,
        tokenlight_moge_source_dropout_prob=args.tokenlight_moge_source_dropout_prob,
        tokenlight_moge_source_dropout_region_fraction=args.tokenlight_moge_source_dropout_region_fraction,
    )


def train_main(argv: Sequence[str]) -> None:
    args, raw_config, merged_config = parse_training_args(argv)
    args.loss_impl_version = LOSS_IMPL_VERSION
    args.delta_flow_formula = "MSE((v_pred_j-v_pred_i)-(v_target_j-v_target_i))"
    args.delta_pair_layout = "pairs_first_then_singletons"
    args.delta_noise_sharing = "within_pair"
    args.moge_pointmap_schema = POINTMAP_SCHEMA
    args.moge_input_contract = (
        "cached points_cv+valid; separate clean point-XYZ and point-to-light-direction token streams; "
        "pair-shared random source-token cutout"
    )
    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
        "dataloader_config": accelerate.DataLoaderConfiguration(split_batches=False, even_batches=True),
    }
    if args.train_mode == "single":
        accelerator_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        accelerator_kwargs["mixed_precision"] = args.mixed_precision
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    base.save_training_config_snapshot(args, raw_config, merged_config, accelerator)
    dataset = safe.build_safe_dataset(args)
    dataset = PointMapCachedDataset(
        dataset,
        cache_root=args.tokenlight_moge_cache_root,
        dataset_root=args.dataset_base_path,
    )
    model = _build_model(args, accelerator)
    model_logger = base._make_model_logger(args)
    delta.launch_delta_training_task(accelerator, dataset, model, model_logger, args=args)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: train_moge3_pointmap.py {precompute|verify-cache|train} ...\n"
            "  precompute    cache MoGe-3 points_cv + valid with torchrun\n"
            "  verify-cache validate cache coverage and shapes\n"
            "  train         TokenLight point/direction streams + delta-flow lambda=0.2"
        )
        return
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "precompute":
        precompute_main(argv)
    elif command == "verify-cache":
        verify_cache_main(argv)
    elif command == "train":
        train_main(argv)
    else:
        raise SystemExit(f"unknown command: {command!r}")


if __name__ == "__main__":
    main()
