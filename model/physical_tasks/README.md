# Physical-task baselines

LightNet uses only samples whose manifest task is `single_light` and whose
filename starts with `light_`. VisibilityNet and ShadowNet use the `position`
samples and geometry-ray masks from `objaverse_fixed32_png`.

## Individual model files

- `lightnet.py`: inverse lighting from source RGB + target RGB only. It
  predicts canonical camera-to-light direction, log distance, intensity,
  radius, RGB color, and ambient scale.
- `visibilitynet.py`: predicts the fixed32 front-facing, unoccluded object mask.
- `shadownet.py`: predicts only the fixed32 cleaned receiver cast-shadow mask,
  `object_shadow_geometry_ray_clean_minarea00.png`.

Actual network inputs are:

- LightNet: source RGB + target RGB (6 channels), with no PBR image loading.
- VisibilityNet: source RGB + world point/normal + pixel-to-light direction +
  log distance + `N dot L` + object mask (15 channels).
- ShadowNet: the VisibilityNet inputs + receiver mask (16 channels). It never
  receives the target shadow mask as an input.

The fixed32 point maps are reconstructed in world coordinates from metric
depth, camera FOV/location, and the recorded camera-to-world rotation. The
stored world-space normal map and per-sample `light.world_position` therefore
share the same coordinate system.

`objaverse_fixed32_png` provides exact geometry-ray binary masks for the latter
two tasks. It does not provide fractional area-light visibility, first-hit
distance, blocker identity, or hidden-geometry observability, so those extra
heads still require regenerated Blender teacher passes.

## Manifest

```bash
python -m model.physical_tasks.build_manifest \
  --data-roots data/portrait_png data/objaverse_test_seen data/objaverse_test_unseen \
  --output data_train/physical_tasks_single_light/metadata.jsonl
```

Visibility/shadow manifest:

```bash
python -m model.physical_tasks.build_fixed32_manifest \
  --data-root data/objaverse_fixed32_png \
  --output data_train/objaverse_fixed32_physical/metadata.jsonl
```

The default split is deterministic by dataset/scene, not by image, so images
from one scene never straddle train and validation.

## Train

Select the physical GPU before launching. Inside the process it appears as
`cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=7 python -m model.physical_tasks.train_lightnet \
  --manifest data_train/physical_tasks_single_light/metadata.jsonl \
  --output-dir outputs/physical_tasks/lightnet_single_light \
  --image-size 480 --batch-size 16 --epochs 50
```

```bash
CUDA_VISIBLE_DEVICES=7 python -m model.physical_tasks.train_visibilitynet \
  --manifest data_train/objaverse_fixed32_physical/metadata.jsonl \
  --output-dir outputs/physical_tasks/visibilitynet_fixed32 \
  --image-size 480 --batch-size 4 --epochs 50
```

```bash
CUDA_VISIBLE_DEVICES=7 python -m model.physical_tasks.train_shadownet \
  --manifest data_train/objaverse_fixed32_physical/metadata.jsonl \
  --output-dir outputs/physical_tasks/shadownet_fixed32 \
  --image-size 480 --batch-size 4 --epochs 50
```

These commands train on each manifest's scene-level `train` split and evaluate
the `val` split every epoch. `--split all` fits every row without held-out
validation: 7,832 LightNet rows or 35,614 accepted fixed32 mask rows.
