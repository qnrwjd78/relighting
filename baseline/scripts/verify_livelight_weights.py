#!/usr/bin/env python3
"""Verify the exact LiveLight and third-party files used by Stage 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "LiveLight/denoising_unet-110000.pth": (
        5_177_738_918,
        "e3394d393cacbad69433ee71cfacd438dc0a018580e7618cdbe9c36ebfef3026",
    ),
    "LiveLight/reference_unet-110000.pth": (
        3_438_325_026,
        "0c082cb98076237ca9acfe0e140b95e711dd188a6c6b36659b4bfaaed34759f9",
    ),
    "LiveLight/light_guider-110000.pth": (
        4_357_014,
        "2177f0955c792581f044eaefca07fb6bd30b437b26b8229ed1572816d5495a73",
    ),
    "LiveLight/temporal_module-20000.pth": (
        1_817_903_610,
        "5b5da9917e1e4ddccbead7b0a706875109b2b6fb4b13e8b5d2afb1291de788e3",
    ),
    "sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin": (
        3_438_354_725,
        "ee23e3368e4e7c0e4ef636ed61923609c97fcaa583f8bb416e3e0986d4a0cfc6",
    ),
    "sd-image-variations-diffusers/image_encoder/pytorch_model.bin": (
        1_215_993_967,
        "89d2aa29b5fdf64f3ad4f45fb4227ea98bc45156bbae673b85be1af7783dbabb",
    ),
    "sd-vae-ft-mse/diffusion_pytorch_model.bin": (
        334_707_217,
        "1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    checks = []
    all_ok = True
    for relative_path, (expected_size, expected_sha256) in EXPECTED.items():
        path = args.weights_root / relative_path
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha256 = sha256_file(path) if exists and actual_size == expected_size else None
        ok = (
            exists
            and actual_size == expected_size
            and actual_sha256 == expected_sha256
        )
        all_ok = all_ok and ok
        check = {
            "path": str(path.resolve()),
            "exists": exists,
            "expected_size": expected_size,
            "actual_size": actual_size,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "ok": ok,
        }
        checks.append(check)
        print(f"{'OK' if ok else 'FAIL'} {relative_path}", flush=True)

    report = {"all_ok": all_ok, "checks": checks}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
