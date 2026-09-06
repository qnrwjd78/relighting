# Retained training configurations

Only exp_0, exp_1, and exp_2 training configurations remain in this directory.
Other configurations are preserved under `local_archive/other_training/configs/`.
Always pass `--config` explicitly; historical defaults in shared trainers can
refer to older configurations that are no longer in the active checkout.

| Group / variant | Entrypoint | Configuration |
| --- | --- | --- |
| exp_0 RGB baseline | `model/train_single.py` | `rgb_baseline_15ep_b8_ga40.json` |
| exp_0 decoder loss | `model/train_decoder_safe.py` | `rgb_decoder_loss_15ep_b8_ga40.json` |
| exp_0 shadow mask | `model/train_decoder_safe.py` | `rgb_shadow_mask_vae_15ep_b8_ga40.json` |
| exp_0 shadow + light mask | `model/train_decoder_safe.py` | `rgb_shadow_light_mask_vae_15ep_b8_ga40.json` |
| exp_0 delta-flow | `model/train_delta_flow.py` | Baseline config plus delta-flow CLI options / saved run config |
| exp_0 MoGe3 pointmap | `model/train_moge3_pointmap.py train` | Saved run config plus prepared MoGe cache |
| exp_1 baseline / delta | `model/train_scene_cache_v2_retained.py baseline` / `delta` | Corresponding `exp1_*baseline*` / `exp1_*delta*` JSON |
| exp_1 pointmap | `model/train_pointmap_scene_cache_v2.py` | `exp1_*pointmap_baseline*` JSON |
| exp_1 shadow mask | `model/train_scene_cache_shadow_safe_retained.py` | `exp1_*shadow_mask_vae*` JSON |
| exp_2 joint RGB / shadow | `model/train_joint_mask.py` | `exp2_*joint_shadow_mask*` JSON |

The older `model/train_scene_cache_v2.py` entrypoint also remains
because the retained-checkpoint wrappers depend on it. MoGe3 and delta-flow are
part of the actual exp_0 / exp_1 runs, even though their filenames lack `exp`.

Use the existing `scripts/launch_exp1_*` and `scripts/launch_exp2_*` launchers for
their configured GPU counts and batch sizes. Configurations contain local input,
cache, output, and sometimes resume-checkpoint paths: ensure they exist before
launching. Saved run configurations remain local under `outputs/train/exp_*/`.

shadow_c2f uses the command-line options of `scripts/train_shadow_c2f.py` and
`scripts/run_shadow_c2f_pipeline.sh`; see
[its pipeline documentation](../../docs/SHADOW_C2F_PIPELINE.md).
