#!/usr/bin/env python3
"""Train the existing TokenLight objectives from wan_vae_scene_latent_cache_v2 files.

This is an isolated adapter: the baseline and delta-flow implementations are
imported unchanged, while their dataset builder is replaced with a reader for
the self-contained per-scene ``.pt`` cache format.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import train_tokenlight as base  # noqa: E402
from model import train_tokenlight_decoder_safe as safe  # noqa: E402
from model import train_tokenlight_delta_flow as delta  # noqa: E402
from model.lightoken_encoder import parse_attrs_json  # noqa: E402


CACHE_SCHEMA = "wan_vae_scene_latent_cache_v2"
SCENE_SCHEMA = "wan_vae_scene_latent_cache_v2"
DEFAULT_PROMPT = "photorealistic object relighting, preserve geometry and materials"
SCENE_SAMPLES_PER_EPOCH = 64


def _scene_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("scene_id") or row.get("scene_folder") or "").strip()
    if not value:
        raise ValueError("Scene-quota sampling requires scene_id or scene_folder on every row")
    return value


def _scene_pools(rows: list[dict[str, Any]], quota: int) -> dict[str, tuple[int, ...]]:
    pools: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pools[_scene_id(row)].append(index)
    return {scene: tuple(indices) for scene, indices in sorted(pools.items())}


class SceneQuotaBatchSampler:
    """Select up to 64 fresh rows per scene on every epoch, without replacement."""

    def __init__(self, rows, task_batch: dict[str, int], *, seed: int = 0) -> None:
        self.rows = rows
        self.seed = int(seed)
        self.quota = SCENE_SAMPLES_PER_EPOCH
        self.batch_size = int(sum(int(value) for value in task_batch.values()))
        if self.batch_size <= 0:
            raise ValueError("Scene-quota batch size must be positive")
        self.pools = _scene_pools(rows, self.quota)
        self.scene_counts = {
            scene: min(self.quota, len(indices)) for scene, indices in self.pools.items()
        }
        self.selected_rows = sum(self.scene_counts.values())
        if self.selected_rows % self.batch_size:
            raise ValueError(
                f"Scene-quota rows must divide batch size: {self.selected_rows} % {self.batch_size}"
            )
        self._length = self.selected_rows // self.batch_size
        self.epoch = 0
        print(
            "Scene-quota sampler: "
            f"scenes={len(self.pools)}, max_samples_per_scene={self.quota}, "
            f"short_scenes={sum(count < self.quota for count in self.scene_counts.values())}, "
            f"rows_per_epoch={self.selected_rows}, batch_size={self.batch_size}, "
            f"batches={self._length}, seed={self.seed}"
        )

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        epoch = int(self.epoch)
        rng = random.Random(self.seed + epoch)
        selected: list[int] = []
        for scene, indices in self.pools.items():
            selected.extend(rng.sample(indices, self.scene_counts[scene]))
        rng.shuffle(selected)
        if len(selected) != self.selected_rows or len(set(selected)) != self.selected_rows:
            raise RuntimeError("Scene-quota sampler produced duplicate or missing rows")
        self.epoch = epoch + 1
        for start in range(0, len(selected), self.batch_size):
            yield selected[start : start + self.batch_size]


class SceneQuotaDeltaBatchSampler:
    """Delta-flow layout over a fresh up-to-64-row subset of every scene."""

    def __init__(
        self,
        rows,
        *,
        dataset_length: int,
        batch_size: int,
        pairs_per_batch: int,
        attrs_key: str,
        seed: int = 0,
    ) -> None:
        if int(dataset_length) != len(rows):
            raise ValueError(
                "Scene-quota delta sampling requires dataset_repeat=1: "
                f"dataset_length={dataset_length}, rows={len(rows)}"
            )
        self.rows = rows
        self.batch_size = int(batch_size)
        self.pairs_per_batch = int(pairs_per_batch)
        self.attrs_key = str(attrs_key)
        self.seed = int(seed)
        self.quota = SCENE_SAMPLES_PER_EPOCH
        self.pools = _scene_pools(rows, self.quota)
        self.scene_counts = {
            scene: min(self.quota, len(indices)) for scene, indices in self.pools.items()
        }
        self.selected_rows = sum(self.scene_counts.values())
        if self.batch_size <= 0 or 2 * self.pairs_per_batch > self.batch_size:
            raise ValueError("Malformed scene-quota delta batch layout")
        if self.selected_rows % self.batch_size:
            raise ValueError(
                f"Scene-quota rows must divide batch size: {self.selected_rows} % {self.batch_size}"
            )
        self._length = self.selected_rows // self.batch_size
        self.pairable_rows = self.selected_rows
        self.epoch = 0
        print(
            "Scene-quota delta sampler: "
            f"scenes={len(self.pools)}, max_samples_per_scene={self.quota}, "
            f"short_scenes={sum(count < self.quota for count in self.scene_counts.values())}, "
            f"rows_per_epoch={self.selected_rows}, pairs_per_batch={self.pairs_per_batch}, "
            f"batches={self._length}, seed={self.seed}"
        )

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        selected_by_scene: list[list[int]] = []
        for scene, indices in self.pools.items():
            scene_count = self.scene_counts[scene]
            selected = rng.sample(indices, scene_count)
            # All rows in these caches differ only by light position within a
            # scene.  Still validate the delta grouping contract explicitly.
            infos = [delta._pair_row_info(self.rows[index], attrs_key=self.attrs_key) for index in selected]
            if any(info is None for info in infos):
                raise ValueError("Malformed light metadata in scene-quota delta subset")
            group_keys = {info.group_key for info in infos if info is not None}
            positions = {info.position for info in infos if info is not None}
            if len(group_keys) != 1 or len(positions) != scene_count:
                raise ValueError("Scene-quota delta subset violates same-condition/different-position pairing")
            rng.shuffle(selected)
            selected_by_scene.append(selected)

        candidate_pairs: list[tuple[int, int]] = []
        for selected in selected_by_scene:
            candidate_pairs.extend(zip(selected[0::2], selected[1::2]))
        rng.shuffle(candidate_pairs)
        required_pairs = self._length * self.pairs_per_batch
        pairs = candidate_pairs[:required_pairs]
        used = {index for pair in pairs for index in pair}
        singleton_count = self.batch_size - 2 * self.pairs_per_batch
        singletons = [index for selected in selected_by_scene for index in selected if index not in used]
        rng.shuffle(singletons)
        required_singletons = self._length * singleton_count
        if len(pairs) != required_pairs or len(singletons) != required_singletons:
            raise RuntimeError(
                "Scene-quota delta schedule is incomplete: "
                f"pairs={len(pairs)}/{required_pairs}, "
                f"singletons={len(singletons)}/{required_singletons}"
            )

        pair_cursor = 0
        singleton_cursor = 0
        for _ in range(self._length):
            batch: list[int] = []
            for _ in range(self.pairs_per_batch):
                batch.extend(pairs[pair_cursor])
                pair_cursor += 1
            batch.extend(singletons[singleton_cursor : singleton_cursor + singleton_count])
            singleton_cursor += singleton_count
            if len(batch) != self.batch_size or len(set(batch)) != self.batch_size:
                raise RuntimeError("Scene-quota delta sampler produced a malformed batch")
            yield batch


def _load_scene(path: Path, *, mmap: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    value = torch.load(path, **kwargs)
    if not isinstance(value, dict) or value.get("schema") != SCENE_SCHEMA:
        raise ValueError(f"Unexpected scene cache schema in {path}: {getattr(value, 'get', lambda *_: None)('schema')!r}")
    samples = value.get("samples")
    sample_latents = value.get("sample_latents")
    source_latent = value.get("source_latent")
    if not isinstance(samples, list) or not isinstance(sample_latents, torch.Tensor):
        raise ValueError(f"Missing samples/sample_latents in {path}")
    if not isinstance(source_latent, torch.Tensor):
        raise ValueError(f"Missing source_latent in {path}")
    if len(samples) != int(sample_latents.shape[0]):
        raise ValueError(f"Sample metadata/latent mismatch in {path}: {len(samples)} != {sample_latents.shape[0]}")
    return value


def _ambient_by_scene(metadata_path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = str(row.get("scene_id") or row.get("scene_folder") or "")
            if not scene_id or scene_id in result:
                continue
            attrs = parse_attrs_json(row.get("attrs_json"))
            result[scene_id] = float(attrs.get("a", 0.0))
    return result


def _attrs_from_sample(sample: Mapping[str, Any], ambient: float) -> str:
    light = sample.get("light")
    if not isinstance(light, Mapping):
        raise ValueError("Scene-cache sample is missing light metadata")
    position = light.get("canonical_position")
    color = light.get("component_color", light.get("render_color", (1.0, 1.0, 1.0)))
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError(f"Malformed canonical_position: {position!r}")
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise ValueError(f"Malformed light color: {color!r}")
    attrs = {
        "a": float(ambient),
        "dg": 0.0,
        "lights": [
            {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "r": float(color[0]),
                "g": float(color[1]),
                "b": float(color[2]),
                "lambda": float(light.get("power_scale", 0.6)),
                "d": float(light.get("canonical_radius", 0.06)),
            }
        ],
        "t": 1.0,
    }
    return json.dumps(attrs, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def prepare_metadata(cache_root: Path, ambient_metadata: Path, output: Path) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    ambient = _ambient_by_scene(ambient_metadata.resolve())
    scene_paths = sorted(cache_root.glob("**/scenes/*.pt"))
    if not scene_paths:
        raise FileNotFoundError(f"No v2 scene cache files found under {cache_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    scene_count = 0
    row_count = 0
    missing_ambient: list[str] = []
    transforms: set[str] = set()
    with temp.open("w", encoding="utf-8") as handle:
        for scene_path in scene_paths:
            cache = _load_scene(scene_path)
            scene_id = str(cache.get("scene_id") or scene_path.stem)
            if scene_id not in ambient:
                missing_ambient.append(scene_id)
                continue
            if int(cache.get("resolution", 0)) != 480:
                raise ValueError(f"Expected 480 cache, got {cache.get('resolution')} in {scene_path}")
            transforms.add(str(cache.get("image_transform")))
            samples = cache["samples"]
            sample_images = cache.get("sample_images")
            if not isinstance(sample_images, list) or len(sample_images) != len(samples):
                raise ValueError(f"Missing or mismatched sample_images in {scene_path}")
            relative_cache = scene_path.relative_to(cache_root).as_posix()
            for sample_index, (sample, sample_image) in enumerate(zip(samples, sample_images)):
                if not isinstance(sample, Mapping):
                    raise ValueError(f"Malformed sample {sample_index} in {scene_path}")
                light = sample.get("light")
                if not isinstance(light, Mapping):
                    raise ValueError(f"Missing light for sample {sample_index} in {scene_path}")
                row = {
                    "scene_id": scene_id,
                    "scene_folder": scene_id,
                    "task": str(sample.get("task") or "position"),
                    "sample_name": str(sample.get("name") or Path(str(sample_image)).stem),
                    "light_id": int(light.get("id", sample_index)),
                    "input_image": f"scenes/{scene_id}/source.png",
                    "video": f"scenes/{scene_id}/{sample_image}",
                    "prompt": DEFAULT_PROMPT,
                    "attrs_json": _attrs_from_sample(sample, ambient[scene_id]),
                    "valid": True,
                    "_scene_cache_file": relative_cache,
                    "_scene_cache_sample_index": sample_index,
                    "_vae_latent_cache_source": relative_cache.split("/", 1)[0],
                }
                handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
                row_count += 1
            scene_count += 1
    if missing_ambient:
        raise KeyError(
            f"Missing ambient attributes for {len(missing_ambient)} cache scenes, first={missing_ambient[:5]}"
        )
    temp.replace(output)
    summary = {
        "schema": CACHE_SCHEMA,
        "cache_root": cache_root.as_posix(),
        "ambient_metadata": ambient_metadata.resolve().as_posix(),
        "metadata_output": output.resolve().as_posix(),
        "scene_count": scene_count,
        "row_count": row_count,
        "image_transforms": sorted(transforms),
    }
    output.with_name("metadata_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


class WanVaeSceneCacheV2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path: Path,
        cache_root: Path,
        *,
        height: int,
        width: int,
        repeat: int = 1,
        max_open_scenes: int = 8,
    ) -> None:
        self.metadata_path = metadata_path.resolve()
        self.cache_root = cache_root.resolve()
        self.height = int(height)
        self.width = int(width)
        self.repeat = int(repeat)
        self.max_open_scenes = max(1, int(max_open_scenes))
        self.load_from_cache = False
        self.skip_tokenlight_file_filter = True
        self.is_vae_latent_cache_dataset = True
        self.is_scene_latent_cache_dataset = True
        self._open: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.data = base._read_jsonl_rows(self.metadata_path)
        if not self.data:
            raise ValueError(f"No rows in prepared scene-cache metadata: {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.data) * self.repeat

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_open"] = OrderedDict()
        return state

    def _scene(self, relative: str) -> dict[str, Any]:
        cached = self._open.get(relative)
        if cached is not None:
            self._open.move_to_end(relative)
            return cached
        path = self.cache_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        cached = _load_scene(path)
        self._open[relative] = cached
        self._open.move_to_end(relative)
        while len(self._open) > self.max_open_scenes:
            self._open.popitem(last=False)
        return cached

    def _make_item(self, index: int) -> dict[str, Any]:
        row = dict(self.data[index % len(self.data)])
        relative = str(row["_scene_cache_file"])
        sample_index = int(row["_scene_cache_sample_index"])
        cache = self._scene(relative)
        row["_tokenlight_cached_latents"] = True
        row["_tokenlight_height"] = self.height
        row["_tokenlight_width"] = self.width
        row["_tokenlight_input_latents"] = cache["sample_latents"][sample_index].detach().cpu().contiguous()
        row["_tokenlight_source_latents"] = cache["source_latent"].detach().cpu().contiguous()
        return row

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._make_item(int(index))

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self._make_item(int(index)) for index in indices]


def build_scene_cache_dataset(args) -> WanVaeSceneCacheV2Dataset:
    return WanVaeSceneCacheV2Dataset(
        base._resolve_path(args.dataset_metadata_path, base_path=ROOT),
        base._resolve_path(args.dataset_base_path, base_path=ROOT),
        height=args.height,
        width=args.width,
        repeat=args.dataset_repeat,
        max_open_scenes=min(8, int(getattr(args, "vae_latent_cache_shard_lru", 8))),
    )


def prepare_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Prepare metadata for wan_vae_scene_latent_cache_v2 training")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--ambient-metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = prepare_metadata(Path(args.cache_root), Path(args.ambient_metadata), Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"prepare", "baseline", "delta"}:
        raise SystemExit("usage: train_tokenlight_scene_cache_v2.py {prepare|baseline|delta} ...")
    mode = sys.argv[1]
    if mode == "prepare":
        prepare_main(sys.argv[2:])
        return
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    base.build_dataset = build_scene_cache_dataset
    base.BalancedTaskBatchSampler = SceneQuotaBatchSampler
    delta.MixedSameScenePairBatchSampler = SceneQuotaDeltaBatchSampler
    if mode == "baseline":
        base.main(train_mode="single")
    else:
        delta.main()


if __name__ == "__main__":
    main()
