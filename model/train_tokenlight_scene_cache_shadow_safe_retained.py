#!/usr/bin/env python3
"""8-GPU-safe shadow-mask conditioning over the exp_1 scene latent cache.

This adapter deliberately composes the existing implementations:

* RGB source/target latents: ``wan_vae_scene_latent_cache_v2``
* shadow condition latents: the safe trainer's separate VAE cache wrapper
* objective/model: the existing latent-only ``rgb_shadow_mask_vae`` path
* sampling: up to 64 positions per scene, trimmed to complete global batches
* checkpoints: one-based names with every fifth epoch retained
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import train_tokenlight as base  # noqa: E402
from model import train_tokenlight_decoder_safe as safe  # noqa: E402
from model import train_tokenlight_scene_cache_v2 as scene_cache  # noqa: E402
from model.train_tokenlight_scene_cache_v2_retained import (  # noqa: E402
    OneBasedRetainedModelLogger,
)


def _scene_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("scene_id") or row.get("scene_folder") or "").strip()
    if not value:
        raise ValueError("Scene-quota sampling requires a scene identity on every row")
    return value


class DistributedSceneQuotaBatchSampler:
    """Sample up to 64 rows per scene and emit complete distributed global batches."""

    def __init__(self, rows, task_batch: dict[str, int], *, seed: int = 0) -> None:
        self.rows = rows
        self.seed = int(seed)
        self.quota = int(scene_cache.SCENE_SAMPLES_PER_EPOCH)
        self.batch_size = sum(int(value) for value in task_batch.values())
        self.world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        if self.batch_size <= 0:
            raise ValueError("Scene-quota batch size must be positive")
        pools: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            pools[_scene_id(row)].append(index)
        self.pools = {key: tuple(value) for key, value in sorted(pools.items())}
        self.scene_counts = {
            scene_id: min(self.quota, len(indices))
            for scene_id, indices in self.pools.items()
        }
        self.available_rows = sum(self.scene_counts.values())
        global_multiple = self.batch_size * self.world_size
        self.selected_rows = self.available_rows - self.available_rows % global_multiple
        self.dropped_rows = self.available_rows - self.selected_rows
        if self.selected_rows <= 0:
            raise ValueError(
                f"No complete global batch: rows={self.available_rows}, multiple={global_multiple}"
            )
        self._length = self.selected_rows // self.batch_size
        self.epoch = 0
        print(
            "Distributed scene-quota sampler: "
            f"scenes={len(self.pools)}, max_samples_per_scene={self.quota}, "
            f"short_scenes={sum(count < self.quota for count in self.scene_counts.values())}, "
            f"available_rows={self.available_rows}, selected_rows={self.selected_rows}, "
            f"dropped_rows={self.dropped_rows}, per_rank_batch={self.batch_size}, "
            f"world_size={self.world_size}, batches={self._length}, seed={self.seed}",
            flush=True,
        )

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        epoch = int(self.epoch)
        rng = random.Random(self.seed + epoch)
        selected: list[int] = []
        for scene_id, indices in self.pools.items():
            selected.extend(rng.sample(indices, self.scene_counts[scene_id]))
        if len(selected) != self.available_rows or len(set(selected)) != self.available_rows:
            raise RuntimeError("Scene-quota sampler produced duplicate or missing rows")
        rng.shuffle(selected)
        selected = selected[: self.selected_rows]
        self.epoch = epoch + 1
        for start in range(0, self.selected_rows, self.batch_size):
            yield selected[start : start + self.batch_size]


def main() -> None:
    # safe.build_safe_dataset calls this shared base function first, then wraps
    # the scene cache with _MaskLatentCacheDataset from the safe trainer.
    base.build_dataset = scene_cache.build_scene_cache_dataset
    base.BalancedTaskBatchSampler = DistributedSceneQuotaBatchSampler
    base.ModelLogger = OneBasedRetainedModelLogger
    safe.main()


if __name__ == "__main__":
    main()
