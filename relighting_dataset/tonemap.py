from __future__ import annotations

import numpy as np


def reinhard(linear):
    image = np.asarray(linear, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image, 0.0)
    return image / (1.0 + image)


def to_uint8(image, gamma: float = 2.2):
    srgb = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if gamma > 0:
        srgb = srgb ** (1.0 / float(gamma))
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

