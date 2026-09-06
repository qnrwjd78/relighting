#!/usr/bin/env python3
"""Scene-cache launcher with one-based, milestone-preserving checkpoints.

This wrapper intentionally leaves the existing trainers untouched.  Epochs are
written as 1..N.  Every fifth epoch is retained permanently, while only the
newest non-milestone epoch is kept.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import train as base  # noqa: E402
from model import train_scene_cache_v2 as scene_cache  # noqa: E402


_EPOCH_CHECKPOINT = re.compile(r"^epoch-(\d+)\.safetensors$")


class OneBasedRetainedModelLogger(base.ModelLogger):
    """Save one-based epoch names and prune obsolete non-milestones."""

    def on_epoch_end(self, accelerator, model, epoch_id):
        saved_epoch = int(epoch_id) + 1
        super().on_epoch_end(accelerator, model, saved_epoch)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            removed = self.prune_epoch_checkpoints(Path(self.output_path), saved_epoch)
            kept = sorted(
                path.name for path in Path(self.output_path).glob("epoch-*.safetensors")
            )
            print(
                f"Checkpoint retention: saved=epoch-{saved_epoch}.safetensors, "
                f"removed={removed}, kept={kept}",
                flush=True,
            )
        accelerator.wait_for_everyone()

    @staticmethod
    def prune_epoch_checkpoints(output_path: Path, newest_epoch: int) -> list[str]:
        """Keep all multiples of five plus the newest epoch checkpoint."""

        output_path = output_path.resolve()
        if not output_path.is_dir():
            return []
        removed: list[str] = []
        for path in output_path.glob("epoch-*.safetensors"):
            match = _EPOCH_CHECKPOINT.fullmatch(path.name)
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch == int(newest_epoch) or epoch % 5 == 0:
                continue
            path.unlink()
            removed.append(path.name)
        return sorted(removed)


def main() -> None:
    base.ModelLogger = OneBasedRetainedModelLogger
    scene_cache.main()


if __name__ == "__main__":
    main()
