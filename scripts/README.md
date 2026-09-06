# Scripts for exp_0, exp_1, exp_2, and shadow_c2f

These scripts retain their original paths so imports, shell launchers, and
pipeline calls continue to work. Run them from the project root.

## Entry points

| Workflow | Start here |
| --- | --- |
| exp_0 inference / evaluation | `infer_exp0.py` |
| exp_1 shadow-mask training | `launch_exp1_shadow_mask_8gpu.sh` |
| exp_2 joint training | `launch_exp2_joint_rgb_shadow_mask_4gpu.sh` |
| exp_2 mask inference | `infer_joint_shadow_mask.py` |
| RGB baseline / delta / pointmap evaluation | `run_objaverse245_rgb_random3_eval_pipeline.sh` |
| shadow_c2f full pipeline | `run_shadow_c2f_pipeline.sh --stage all --dry-run` |

For exp_0 and exp_1 baseline/delta/pointmap training, use the retained `model/`
entrypoints listed in the [configuration guide](../configs/train_480/README.md).

## Retained files and purpose

### exp_0 inference and seed comparison

- [infer_exp0.py](infer_exp0.py)
- [infer_seed_sweep.py](infer_seed_sweep.py)

### shared RGB, delta, and pointmap inference

- [infer_manifest.py](infer_manifest.py)
- [infer_manifest_moge3_pointmap.py](infer_manifest_moge3_pointmap.py)
- [infer_manifest_scene_cache_v2.py](infer_manifest_scene_cache_v2.py)
- [infer_manifest_pointmap_scene_cache_v2.py](infer_manifest_pointmap_scene_cache_v2.py)
- [infer_manifest_pointmap_rgb_npy.py](infer_manifest_pointmap_rgb_npy.py)

### exp_1 preparation and launch

- [decode_train_object_masks_from_vae_cache.py](decode_train_object_masks_from_vae_cache.py)
- [prepare_exp1_shadow_mask_metadata.py](prepare_exp1_shadow_mask_metadata.py)
- [prepare_exp1_shadow_mask_training.sh](prepare_exp1_shadow_mask_training.sh)
- [preflight_exp1_shadow_mask_training.py](preflight_exp1_shadow_mask_training.py)
- [launch_exp1_shadow_mask_8gpu.sh](launch_exp1_shadow_mask_8gpu.sh)
- [launch_exp1_shadow_mask_8gpu_b4_nogc.sh](launch_exp1_shadow_mask_8gpu_b4_nogc.sh)
- [launch_exp1_shadow_mask_8gpu_nogc.sh](launch_exp1_shadow_mask_8gpu_nogc.sh)

### exp_2 launch and inference

- [launch_exp2_joint_rgb_shadow_mask_4gpu.sh](launch_exp2_joint_rgb_shadow_mask_4gpu.sh)
- [launch_exp2_joint_rgb_shadow_mask_8gpu.sh](launch_exp2_joint_rgb_shadow_mask_8gpu.sh)
- [launch_exp2_joint_rgb_shadow_mask_resume_e3_gpu4567.sh](launch_exp2_joint_rgb_shadow_mask_resume_e3_gpu4567.sh)
- [infer_joint_shadow_mask.py](infer_joint_shadow_mask.py)

### RGB evaluation and its video dependencies

- [prepare_objaverse245_eval.py](prepare_objaverse245_eval.py)
- [select_random_scene_heights.py](select_random_scene_heights.py)
- [run_objaverse245_rgb_random3_eval_pipeline.sh](run_objaverse245_rgb_random3_eval_pipeline.sh)
- [summarize_metrics_by_height.py](summarize_metrics_by_height.py)
- [make_height_output_gt_videos_rgb.py](make_height_output_gt_videos_rgb.py)
- [make_height_output_gt_videos.py](make_height_output_gt_videos.py)

### shadow_c2f preparation, training, inference, and diagnostics

- [build_shadow_physics_cache.py](build_shadow_physics_cache.py)
- [decode_shadow_c2f_train_rgb.py](decode_shadow_c2f_train_rgb.py)
- [infer_shadow_c2f.py](infer_shadow_c2f.py)
- [infer_shadow_c2f_online_trajectory.py](infer_shadow_c2f_online_trajectory.py)
- [preflight_shadow_c2f_pipeline.py](preflight_shadow_c2f_pipeline.py)
- [prepare_shadow_c2f_manifest.py](prepare_shadow_c2f_manifest.py)
- [prepare_shadow_c2f_online_manifest.py](prepare_shadow_c2f_online_manifest.py)
- [run_shadow_c2f_pipeline.sh](run_shadow_c2f_pipeline.sh)
- [run_shadow_c2f_wan_eval.sh](run_shadow_c2f_wan_eval.sh)
- [run_shadowadapter_cache.py](run_shadowadapter_cache.py)
- [train_shadow_c2f.py](train_shadow_c2f.py)
- [visualize_shadowadapter_cache.py](visualize_shadowadapter_cache.py)

### shadow_c2f scene 2583 trajectory and its producer

- [prepare_scene2583_height09_c2f.py](prepare_scene2583_height09_c2f.py)
- [build_scene2583_height09_external.py](build_scene2583_height09_external.py)

## Dependencies that are easy to overlook

- `infer_exp0.py` and `infer_seed_sweep.py` invoke `infer_manifest.py`.
- NPY pointmap inference imports both the MoGe3 and scene-cache inference helpers.
- The RGB evaluation shell script invokes the height metric summary and RGB
  video builder; that builder imports `make_height_output_gt_videos.py`.
- `prepare_scene2583_height09_c2f.py` reads `trajectory_manifest.json` generated
  by `build_scene2583_height09_external.py`. The latter is kept despite its name
  and its additional external-comparison outputs. It does not run those models.
- AdapterShadow remains necessary for cached shadow_c2f; FOCUS remains necessary
  for its online path. Their local repositories and weights are still excluded
  from Git. See [the pipeline contract](../docs/SHADOW_C2F_PIPELINE.md).

## Archived files

Unrelated comparison tools, PBR/CoShadow/LGI/GT-mask inference scripts, their
configuration files, and standalone unrelated utilities were moved intact to
`local_archive/other_scripts/`, preserving relative paths. The now-unused
`utils/spatial_condition_reader.py` is stored there as well. The directory is
ignored by Git. Its `manifest.json` records original paths and SHA-256 digests.
Restore archived files to their original paths before using them; do not run
them directly from the archive because many depend on their directory depth.

## Verification

After archiving, all 28 Python scripts and 11 retained trainers imported, 26
script CLI help checks passed, all 10 shell scripts passed syntax checks, and
49 CPU regression tests passed. The two fixed-scene builders have no help CLI;
they were checked by parsing and importing, without generating local data.
The shadow pipeline dry-run also resolved all of its script entrypoints.

The scene-cache inference wrappers require `--scene-cache-root` even with
`--help`. For example:

```bash
python scripts/infer_manifest_scene_cache_v2.py --scene-cache-root data/vae_cache_480 --help
```

Direct execution now resolves project imports in the online shadow inference
and luminance video scripts. `matplotlib`, needed by their evaluation helpers,
is included in `docker/requirements.txt`.
These checks do not constitute full GPU training or real-data inference.

## Shared training launcher implementation

The six `launch_exp1_*` / `launch_exp2_*` scripts keep their public filenames but
now delegate repeated preflight and Accelerate commands to
`lib/training_launch.sh`. Each wrapper retains its configuration, batch checks,
GPU selection, allocator setting, and resume-checkpoint requirement. A failed
preflight still stops training. Repository paths are resolved relative to the
launcher instead of assuming the checkout itself is at `/workspace`.

Original launchers are backed up under `local_archive/model_cleanup/scripts/`.
`PYTHON_BIN` and `ACCELERATE_BIN` optionally override executable paths; defaults
resolve `python` and `accelerate` from the activated Conda environment. Command-capture tests verify all six variants,
preflight failure, and resume-checkpoint checks without starting training.
The suite now contains 53 passing CPU tests.

## Pretrained weights

`download_weights.sh` downloads all public base weights, with `--only` to select
one group and `--dry-run` / `--check` for planning and offline checks. See the
[Conda setup guide](../docs/CONDA_SETUP.md) for the separate AdapterShadow SBU
checkpoint input. Missing SBU results in nonzero exit after other downloads.

## GenLit and LiveLight comparison tools

The 20 comparison scripts and JSON/YAML configurations are in
[`baseline/scripts/`](../baseline/scripts/), with their source repositories in
`baseline/repos/`. See the [baseline guide](../baseline/README.md) for setup,
download, and run commands. Weights remain in the root `weights/` directory.
The shared `scripts/download_weights.sh` covers the core exp_0/1/2 and shadow_c2f
weights; baseline downloads use the model-specific scripts.
