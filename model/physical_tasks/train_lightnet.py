#!/usr/bin/env python3
"""Dedicated LightNet training entrypoint."""

from __future__ import annotations

from .train import main


if __name__ == "__main__":
    raise SystemExit(main(forced_model="lightnet"))
