#!/usr/bin/env python3
"""Create a GenLit multi trajectory copy with a chosen point-light intensity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--point-intensity", required=True, type=float)
    args = parser.parse_args()

    trajectories = np.load(args.input, allow_pickle=True)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (25, 3):
        raise ValueError(
            f"Expected GenLit multi manifest [N,25,3], got {trajectories.shape}"
        )

    # Preserve the first two columns (position and environment intensity) and only
    # replace the physical point-light intensity consumed by the fifth channel.
    output = trajectories.astype(object, copy=True)
    output[:, :, 2] = format(args.point_intensity, ".12g")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(f"output={args.output}")
    print(f"shape={output.shape}")
    print(f"point_intensity={args.point_intensity:.12g}")


if __name__ == "__main__":
    main()
