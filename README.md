# Relighting: exp_0, exp_1, exp_2, and shadow_c2f

This checkout keeps the training entrypoints for these four experiment groups
and the shared modules needed by their training, inference, and evaluation.
Other training code is stored locally in `local_archive/other_training/`.
Unrelated scripts are stored in `local_archive/other_scripts/`. Git ignores both.
See the [script guide](scripts/README.md) for the retained tools and dependencies.

## Layout

- `model/`: retained TokenLight trainers and shared model components; see
  [the model guide](model/README.md).
- `model/shadow_c2f/`: shadow refinement network, geometry, loss, and metrics.
- `scripts/`: launchers, preprocessing, inference, and evaluation.
- `utils/`, `relighting_dataset/`: shared data and evaluation utilities.
- `configs/train_480/`: exp_0, exp_1, and exp_2 experiment configurations.
- `tests/`: regression tests for the retained workflows.
- `docs/`: pipeline documentation and historical experiment notes.
- `docker/`: CUDA image and dependency definition.

Datasets, weights, caches, outputs, environments, third-party checkouts, and the
local archive are excluded from Git. See [repository notes](docs/REPOSITORY.md)
and [retained training scope](docs/TRAINING_SCOPE.md).

## Environment

Use [the Conda setup guide](docs/CONDA_SETUP.md) for a new machine:

```bash
conda env create -f environment.yml
conda activate relighting
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r docker/requirements.txt
bash scripts/download_weights.sh --only wan
```

The requirements file is shared with the optional Docker image; Conda setup
requires no Docker. Connect your existing data and manifest directories and
prepare any missing caches. Some experiment paths still assume `/workspace`.
The setup guide distinguishes official base weights from your own trained
checkpoints and lists the optional shadow_c2f dependencies.

## Training

See [the configuration guide](configs/train_480/README.md) for the entrypoints
and variants. Examples:

```bash
# exp_0 baseline
python model/train_single.py \
  --config configs/train_480/rgb_baseline_15ep_b8_ga40.json

# exp_1 RGB baseline with scene-cache sampling and retained checkpoints
python model/train_scene_cache_v2_retained.py baseline \
  --config configs/train_480/exp1_7x7x5_power06_rgb_baseline_scene64_fresh_retained.json

# exp_1 shadow mask / exp_2 joint RGB + shadow mask (multi-GPU)
bash scripts/launch_exp1_shadow_mask_8gpu.sh
bash scripts/launch_exp2_joint_rgb_shadow_mask_4gpu.sh

# shadow_c2f: inspect required training inputs and preview the pipeline
python scripts/train_shadow_c2f.py --help
bash scripts/run_shadow_c2f_pipeline.sh --stage all --dry-run
```

The exp_2 resume launcher requires the exact checkpoint configured in its JSON.
See [the shadow pipeline contract](docs/SHADOW_C2F_PIPELINE.md) for external
models, cache preparation, training, and inference prerequisites.

## Inference and evaluation

```bash
python scripts/infer_exp0.py --help
python scripts/infer_manifest_scene_cache_v2.py --scene-cache-root data/vae_cache_480 --help
python scripts/infer_joint_shadow_mask.py --help
python scripts/infer_shadow_c2f.py --help
```

Shared inference components required by these experiments are kept. Unrelated
PBR, CoShadow, LGI, GT-mask, GenLit, and LiveLight scripts are archived. The
shadow_c2f AdapterShadow and online FOCUS paths remain available.

## Verification

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

Historical documents may mention archived training paths. Those commands require
restoring the corresponding files from the local archive first; they are outside
the supported training scope of this checkout.
