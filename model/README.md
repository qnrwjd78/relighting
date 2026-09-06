# Model code for exp_0, exp_1, exp_2, and shadow_c2f

The root contains 18 Python files. Their existing paths are preserved because
trainers, inference scripts, and saved experiment commands import them directly.
The shadow refiner is already isolated in `shadow_c2f/`.

## Where to start

| Purpose | Files |
| --- | --- |
| exp_0 baseline / single GPU / ZeRO-3 | `train_tokenlight.py`, `train_tokenlight_single.py`, `train_tokenlight_zero3.py` |
| exp_0 decoder and delta-flow variants | `train_tokenlight_decoder_safe.py`, `train_tokenlight_delta_flow.py` |
| exp_0 MoGe3 point/direction variant | `train_tokenlight_moge3_pointmap.py` |
| exp_1 scene-cache baseline / delta | `train_tokenlight_scene_cache_v2.py`, `train_tokenlight_scene_cache_v2_retained.py` |
| exp_1 NPY pointmap | `train_tokenlight_pointmap_scene_cache_v2.py` |
| exp_1 shadow conditioning | `train_tokenlight_scene_cache_shadow_safe_retained.py` |
| exp_2 joint RGB / shadow | `train_tokenlight_joint_mask.py`, `tokenlight_joint_mask.py` |
| shadow_c2f network / loss / geometry | `shadow_c2f/`; entrypoint: `../scripts/train_shadow_c2f.py` |
| Shared inference / Wan weights | `infer_tokenlight.py`, `pretrain_weight.py` |
| Shared light tokens and Wan integration | `lightoken_encoder.py`, `tokenlight_wan.py` |
| Pointmap clean-prefix timestep helper | `tokenlight_wan_spatial.py` |
| Illumination transformations used by latent caching | `illumination_latent_head.py` |

See [training configurations](../configs/train_480/README.md) and
[script entrypoints](../scripts/README.md) for commands.

## Similar filenames do not imply duplicate behavior

The Python files were compared by content hash and parsed syntax tree; none were
exact duplicates. Several related files deliberately implement different contracts:

- `single` and `zero3` are 15-line wrappers selecting different training modes.
- `scene_cache_v2_retained` installs one-based epoch numbering and checkpoint
  retention on top of the shared scene-cache trainer.
- `scene_cache_shadow_safe_retained` adds DDP-safe shadow conditioning.
- RGB, delta-flow, decoder loss, joint-mask, and pointmap trainers implement
  different objectives or data inputs and share base modules through imports.
- `tokenlight_wan_spatial.py` is still imported by MoGe3 for clean-prefix timestep
  handling; `illumination_latent_head.py` is still imported by the latent cache
  builder for image transforms. Both remain dependencies of the active workflows.

Keeping these import paths avoids changing runtime monkey-patching, checkpoint
semantics, or the training objective merely to make filenames look simpler.

## Archived code

Ten unrelated or superseded model files were moved intact to
`local_archive/model_cleanup/model/`: the official/standalone Wan examples,
batch and debug inference tools, PBR and CoShadow model extensions, the unused
legacy decoder loss, and the unused physical-training runtime.
The archive is ignored by Git and records original paths and SHA-256 digests in
`local_archive/model_cleanup/manifest.json`. The earlier training archive remains
under `local_archive/other_training/`.

Local `train/` and `Wan-AI/` directories hold outputs or downloaded model assets
and remain ignored. The cleanup does not alter these assets.
