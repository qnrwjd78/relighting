# Conda setup on a new Linux NVIDIA GPU machine

This guide installs the retained TokenLight experiments without Docker. Existing
data can be linked into the checkout; pretrained weights are downloaded again.
The host must already have a working NVIDIA driver (`nvidia-smi`). The commands
below target the current PyTorch 2.6 / CUDA 12.4 stack, not every GPU generation.

## Create the environment

From the cloned repository root:

```bash
conda env create -f environment.yml
conda activate tokenlight
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r docker/requirements.txt
python -m pip check
```

`docker/requirements.txt` is the shared Python dependency list despite its
historical directory name; Docker is not used by these commands. Conda supplies
Python 3.10, FFmpeg, Ninja, a C++ compiler, and pkg-config. Install PyTorch first
so packages with torch-dependent build steps can find it.
The CUDA wheel command is from the [official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/).
The environment specification is not a complete lockfile; some Python packages
remain unpinned. A clean solve/build has not yet been executed on the target host.

CUDA extensions used by FOCUS or selected DeepSpeed optimizers additionally need
a compatible CUDA toolkit with `nvcc` and a correctly set `CUDA_HOME`. PyTorch's
CUDA wheel does not supply the full CUDA compiler toolkit. Install that toolkit
before compiling those extensions; do not assume `nvidia-smi` proves nvcc exists.

```bash
which python accelerate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

Launchers now use `python` and `accelerate` from the activated environment.
Unset old `PYTHON_BIN` / `ACCELERATE_BIN` overrides if they point at another
installation. The host's system Python is not needed for normal training.

## Download all public weights with one script

After activating the Conda environment and installing the requirements:

```bash
bash scripts/download_weights.sh
```

This downloads the pinned Wan snapshot, MoGe3, FOCUS-L shadow detection, SAM
ViT-B, EfficientNet-B1, and FOCUS's CLIP ViT-B/16 backbone. CLIP is placed in
`~/.cache/clip/` because the upstream FOCUS loader uses that directory. All other
weights go to the project paths expected by the existing code. A report is saved
in `weights/download_report.json`. Hugging Face downloads reuse cached files;
direct downloads publish files atomically and check SHA-256. Existing corrupt
files are reported rather than silently overwritten.

AdapterShadow's official distribution ZIP does not contain `sbu.ckpt`. Provide
its separate official checkpoint when available:

```bash
bash scripts/download_weights.sh --sbu-file /path/to/sbu.ckpt
# Alternatively: ADAPTERSHADOW_SBU_URL='https://...' bash scripts/download_weights.sh
```

Without SBU, the script downloads the other groups and then exits nonzero with
an INCOMPLETE report. It does not pretend the AdapterShadow pipeline is ready.
Your trained TokenLight/C2F checkpoints and external model source/installations
are outside this weight download step.

```bash
bash scripts/download_weights.sh --dry-run        # no writes or network
bash scripts/download_weights.sh --only wan       # baseline weights only
bash scripts/download_weights.sh --only focus     # FOCUS-L-SD and CLIP
bash scripts/download_weights.sh --only moge
bash scripts/download_weights.sh --only adapter
bash scripts/download_weights.sh --check          # offline check, no writes
```

Offline checking verifies presence and direct-download hashes. Online Hugging
Face downloads additionally verify expected file sizes and LFS SHA-256 hashes
from the pinned repository metadata. No model loading/unpickling is performed.

## Download Wan weights

Download the native Wan checkpoint layout expected by `model/pretrain_weight.py`:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir weights/Wan2.2-TI2V-5B
python -c "from model.pretrain_weight import validate_wan22_weights; validate_wan22_weights(); print('Wan files OK')"
```

The [official Wan repository](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/main)
contains roughly 34 GB. Its files include three diffusion shards, the text
encoder, VAE, and `google/umt5-xxl` tokenizer. The validator checks file presence;
it is not a full model-load or content-integrity test. Re-running `hf download`
reuses downloaded files; see the [Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download).
Weights and their download metadata stay ignored by Git.

## Connect existing data

If the clone does not already contain `data/` and `data_train/`, link your actual
data and manifest directories (replace the paths below):

```bash
ln -s /your/storage/data data
ln -s /your/storage/data_train data_train
```

Do not overwrite existing directories to create these links. Training also uses
VAE caches, masks, and sometimes pointmap caches; original RGB images alone do
not satisfy those configs. Preserve relative paths under the two data roots.

Some experiment metadata and scripts still contain absolute `/workspace/...`
paths. Use `/workspace` as the checkout path on the new host, or update those
paths to the new checkout. The launch helpers discover the checkout root, but
that does not rewrite paths stored in manifests/configurations.

## Additional weights by workflow

| Workflow | Additional assets |
| --- | --- |
| exp_0 / exp_1 / exp_2 fresh TokenLight training | Wan base weights plus the config's prepared data/caches |
| MoGe pointmap precompute | `Ruicheng/moge-3-vitl` plus the matching MoGe source/runtime; not needed when the required pointmaps are already cached |
| Cached AdapterShadow shadow_c2f | AdapterShadow SBU checkpoint, SAM ViT-B, EfficientNet-B1, and its external code/environment |
| Online Wan → FOCUS → shadow_c2f | A trained baseline TokenLight checkpoint, FOCUS shadow checkpoint/config, Detectron2 and compiled FOCUS ops |
| Existing TokenLight / shadow_c2f inference or resume | Your own trained experiment checkpoint |

For MoGe precompute, the code expects `weights/moge-3-vitl/model.pt` and the
MoGe checkout at `weights/MoGe` (revision is recorded in
`model/train_tokenlight_moge3_pointmap.py`). Download its weight with:

```bash
hf download Ruicheng/moge-3-vitl --local-dir weights/moge-3-vitl
```

The weight alone does not install the MoGe/FlexGEMM runtime. If pointmaps already
exist, use the corresponding cached inference/training path instead.

For AdapterShadow, follow the upstream [model links](https://github.com/LeipingJie/AdapterShadow)
and place the files at:

```text
external/AdapterShadow/checkpoint/sbu.ckpt
external/AdapterShadow/checkpoint/sam/sam_vit_b_01ec64.pth
external/AdapterShadow/efficientnet/hub/checkpoints/tf_efficientnet_b1_ap-44ef0a3d.pth
```

For FOCUS, obtain the shadow-detection checkpoint and matching configuration from
its [model zoo](https://github.com/geshang777/FOCUS/blob/main/MODEL_ZOO.md).
The external repositories/revisions and local FOCUS patch are recorded in
[repository notes](REPOSITORY.md). The base Conda environment does not automatically
install these optional external pipelines. For online FOCUS, its dependencies
must be importable from the same Python process that runs TokenLight/C2F.
For cached AdapterShadow, `ADAPTER_PYTHON_BIN` can select a separate environment.

Official pretrained weights do not contain your trained TokenLight LoRA, light
encoder, or shadow_c2f refiner. Preserve the selected `outputs/train/` checkpoints
if reusing existing results. Otherwise train them again before downstream C2F
conditioning, inference, or resume. A fresh training config must not point to an
unavailable resume checkpoint.

## Verify before training

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
bash scripts/run_shadow_c2f_pipeline.sh --stage all --dry-run
```

The CPU suite and dry-run do not validate real GPU training or installed external
model weights. Check the selected config's GPU count, batch size, data paths,
and checkpoints before using its launcher. In particular, the exp_2 4-GPU
wrappers currently select GPU IDs `4,5,6,7`; use `0,1,2,3` on a four-GPU host.
