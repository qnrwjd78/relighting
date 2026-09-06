from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalize_rgb(image: np.ndarray, path: Path) -> np.ndarray:
    image = np.asarray(image)
    while image.ndim > 3 and 1 in image.shape[:-1]:
        image = np.squeeze(image, axis=next(index for index, size in enumerate(image.shape[:-1]) if size == 1))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[0] in {3, 4} and image.shape[-1] not in {3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"EXR needs HxWx3 RGB data, got shape {image.shape} from {path}")
    return image[..., :3].astype(np.float32, copy=False)


def _read_with_openexr(path: Path) -> np.ndarray:
    import OpenEXR

    try:
        import Imath
    except ImportError:  # pragma: no cover - depends on OpenEXR package version.
        Imath = OpenEXR

    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        data_window = header["dataWindow"]
        width = int(data_window.max.x - data_window.min.x + 1)
        height = int(data_window.max.y - data_window.min.y + 1)
        available = set(header["channels"].keys())
        names = [name for name in ("R", "G", "B") if name in available]
        if len(names) < 3:
            names = [name for name in ("r", "g", "b") if name in available]
        if len(names) < 3:
            names = list(header["channels"].keys())[:3]
        if len(names) < 3:
            raise ValueError(f"EXR needs at least 3 channels, got {sorted(available)}")

        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = [
            np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(height, width)
            for name in names[:3]
        ]
        return np.stack(channels, axis=-1)
    finally:
        close = getattr(exr, "close", None)
        if close is not None:
            close()


def _read_with_imageio(path: Path) -> np.ndarray:
    import imageio.v3 as iio

    return _normalize_rgb(np.asarray(iio.imread(path)), path)


def read_exr(path: str | Path) -> np.ndarray:
    exr_path = Path(path)
    try:
        image = _read_with_openexr(exr_path)
    except ImportError:
        image = _read_with_imageio(exr_path)
    image = _normalize_rgb(np.asarray(image), exr_path)
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
