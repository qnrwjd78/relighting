#!/usr/bin/env python3
"""Dedicated VisibilityNet training entrypoint for objaverse_fixed32_png."""

from __future__ import annotations

from .train_masks import train_main


if __name__ == "__main__":
    raise SystemExit(train_main("visibility"))
