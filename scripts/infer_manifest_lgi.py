#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import infer_manifest_spatial
if __name__ == "__main__":
    if "--spatial-kind" not in sys.argv: sys.argv += ["--spatial-kind", "lgi"]
    raise SystemExit(infer_manifest_spatial.main())
