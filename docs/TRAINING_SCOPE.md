# Retained training scope

The active training scope is exp_0, exp_1, exp_2, and shadow_c2f. Shared
models and the inference, preprocessing, and evaluation tools needed by these
four groups remain in place; see the [script guide](../scripts/README.md).

## Retained entrypoints

- exp_0: `train_tokenlight_single.py`, `train_tokenlight_zero3.py`,
  `train_tokenlight_decoder_safe.py`, `train_tokenlight_delta_flow.py`,
  and `train_tokenlight_moge3_pointmap.py` (under `model/`).
- exp_1: `model/train_tokenlight_scene_cache_v2_retained.py` (baseline/delta),
  `model/train_tokenlight_pointmap_scene_cache_v2.py`, and
  `model/train_tokenlight_scene_cache_shadow_safe_retained.py`.
- exp_2: `model/train_tokenlight_joint_mask.py` and its existing launchers.
- shadow_c2f: `scripts/train_shadow_c2f.py`, `model/shadow_c2f/`, and the
  existing pipeline, inference, cache, and evaluation tools.
- Shared training implementations `model/train_tokenlight.py` and
  `model/train_tokenlight_scene_cache_v2.py` are required by these entrypoints.

MoGe3 / pointmap and delta-flow are present in the local exp_0 / exp_1 run
records and must not be classified as unrelated merely by filename. The original
paths and implementation of every retained trainer are unchanged.

## Local archive

Other training files and their experiment configurations were moved to
`local_archive/other_training/`, preserving their relative paths. This directory
is ignored by Git. `manifest.json` inside it records each original path, archive
path, and SHA-256 digest; all moved files were checked for byte-for-byte identity.
Python bytecode belonging to the archived physical-task package was moved too.
Copies of the previous README/configuration guide and outdated shadow tests are
also kept there for reference.

To restore an archived experiment, copy its required files back to the original
paths recorded in the manifest, without overwriting newer files. The archive
is storage, not an alternate import root or runnable installation. It is not
available in a fresh Git clone.

Tracked files moved out of the active tree appear as deletions in Git; their
local archive copies are ignored. No Git index changes, commits, or pushes were
made. A later commit must include those deletions to remove the old training
files from the repository tip; older commits will still contain them.

### Archived source, configuration, and documentation files

- `configs/train_480/coshadow_box.json`
- `configs/train_480/coshadow_diffusion.json`
- `configs/train_480/fixed32_pbr_depth_normal.json`
- `configs/train_480/pbr.json`
- `configs/train_480/pbr_decoder.json`
- `configs/train_480/rgb.json`
- `configs/train_480/rgb_decoder.json`
- `configs/train_480/rgb_decoder_w_mask.json`
- `configs/train_480/rgb_spatial_gt_masks.json`
- `configs/train_480/rgb_spatial_lgi.json`
- `configs/train_tokenlight_pbr_unirelight_480_cache_allsets_lora128_zero3.json`
- `model/physical_tasks/README.md`
- `model/physical_tasks/__init__.py`
- `model/physical_tasks/build_fixed32_manifest.py`
- `model/physical_tasks/build_manifest.py`
- `model/physical_tasks/data.py`
- `model/physical_tasks/lightnet.py`
- `model/physical_tasks/models.py`
- `model/physical_tasks/shadownet.py`
- `model/physical_tasks/train.py`
- `model/physical_tasks/train_lightnet.py`
- `model/physical_tasks/train_masks.py`
- `model/physical_tasks/train_shadownet.py`
- `model/physical_tasks/train_visibilitynet.py`
- `model/physical_tasks/visibilitynet.py`
- `model/repartition_tokenlight_pbr_zero3.py`
- `model/train.py`
- `model/train_coshadow_box_predictor.py`
- `model/train_tokenlight_coshadow.py`
- `model/train_tokenlight_coshadow_ddp.py`
- `model/train_tokenlight_decoder_20260807.py`
- `model/train_tokenlight_gt_masks.py`
- `model/train_tokenlight_gt_masks_ddp.py`
- `model/train_tokenlight_lgi.py`
- `model/train_tokenlight_lgi_ddp.py`
- `model/train_tokenlight_pbr.py`
- `model/train_tokenlight_pbr_safe.py`
- `model/train_tokenlight_pbr_safe_ddp.py`
- `model/train_tokenlight_pbr_single.py`
- `model/train_tokenlight_pbr_zero3.py`
- `model/train_tokenlight_spatial_safe.py`
- `tests/test_coshadow.py`

## Compatibility and verification

Unrelated scripts were subsequently moved to `local_archive/other_scripts/`.
This includes spatial inference and its extracted condition reader; neither is
needed by the retained experiments. Their archive preserves the original relative
paths. The script guide records the retained entrypoints and indirect dependencies.
Historical design documents may still describe archived paths.

The shadow network already implements `direct_resunet_v2`. Its old tests were
updated to validate the single full-resolution output, one delta input channel,
zero-initialized residual, optional receiver gating, finite gradients, and light
distance sensitivity after a nonzero head is introduced. No network, loss, or
checkpoint behavior was changed.

Validation covers local import resolution, actual trainer/inference imports,
CLI help, shell syntax, the shadow pipeline dry-run, and CPU regression tests.
Full GPU training and real-data inference require the local weights, datasets,
caches, and external checkpoints and are not established by these checks.

## Model cleanup and launcher deduplication

See [the model guide](../model/README.md) for the 18 retained root modules and
why their similar names represent different behavior. Ten unused model files
were moved intact to `local_archive/model_cleanup/model/`. No retained model
implementation changed. Repeated shell launch sequences now live in
`scripts/lib/training_launch.sh`, with original wrapper paths and experiment
settings preserved. The same archive holds the original launcher scripts.
