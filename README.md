# TokenLight Reprod Wan2.2 Baseline

This folder keeps the official DiffSynth-Studio Wan2.2-TI2V-5B baseline and
adds separate TokenLight conditioning entrypoints.

## Layout

```text
model/               baseline, TokenLight, physical tasks, and shadow models
scripts/             training, inference, preprocessing, and evaluation tools
utils/               shared data, geometry, and evaluation helpers
tokenlight_dataset/  EXR loading and tone mapping
configs/             Accelerate / DeepSpeed and experiment configurations
tests/               regression tests
docs/                experiment and pipeline documentation
docker/              CUDA image and Python dependencies
data/, data_train/   local datasets and manifests (not committed)
weights/, outputs/   local model weights and results (not committed)
repos/, external/    local third-party checkouts / packages (not committed)
```

See [repository contents and publishing notes](docs/REPOSITORY.md) for the
Git exclusion policy and local dependencies. Experiment configurations refer to
local datasets, caches, and checkpoints; update these paths before running on a
new machine. Some comparison scripts expect this checkout at `/workspace`.

## Docker

Build from this folder:

```bash
docker build -f docker/Dockerfile -t tokenlight-reprod .
```

Run with the mounted workspace:

```bash
docker run -it --ipc=host --name tokenlight-reprod \
  -v ${PWD}:/workspace --gpus all tokenlight-reprod bash
```

## LoRA Train

The default command follows the official DiffSynth Wan2.2-TI2V-5B LoRA example,
but loads local weights from:

```text
weights/Wan2.2-TI2V-5B
```

Put the DiffSynth example dataset under:

```text
data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B
```

The baseline training entrypoint is:

```text
model/train.py
```

The official DiffSynth inference example is preserved as:

```text
model/infer_official.py
```

Inspect the baseline arguments after installing the dependencies:

```bash
python model/train.py --help
```

For TokenLight training with single GPU or ZeRO-3, use the entrypoints and
experiment configurations in [TokenLight Train](#tokenlight-train) below.

## Inference

Text-to-video:

```bash
python model/infer.py \
  --prompt "Two cute cats wearing boxing gloves fight on a boxing ring."
```

Image-to-video:

```bash
python model/infer.py \
  --input_image data/input.png \
  --prompt "Two cute cats wearing boxing gloves fight on a boxing ring."
```

## TokenLight Metadata

TokenLight training metadata should contain these columns:

```text
video,input_image,mask,prompt,attrs_json
```

Meanings:

```text
video       target relit image/video
input_image source image
mask        optional relighting/object/fixture mask
prompt      fixed generic text prompt
attrs_json  numeric light condition JSON
```

Example `attrs_json`:

```json
{"a":0.014,"x":0.2,"y":-0.4,"z":0.8,"r":1.0,"g":1.0,"b":1.0,"lambda":1.2,"d":0.06}
```

## TokenLight Train

TokenLight training uses source/mask/light prefix tokens before the noisy Wan
target tokens. Text prompt stays fixed; CFG/dropout is applied only to light
tokens.

All 480 training configurations are under `configs/train_480/`:

| Experiment | RGB entrypoint | PBR entrypoint | Config |
| --- | --- | --- | --- |
| RGB latent loss | yes | no | `rgb.json` |
| PBR latent loss | no | yes | `pbr.json` |
| RGB decoder loss | yes | no | `rgb_decoder.json` |
| PBR decoder loss | no | yes | `pbr_decoder.json` |

Single GPU uses the `*_single.py` entrypoint:

```bash
python model/train_tokenlight_single.py --config configs/train_480/rgb.json
python model/train_tokenlight_pbr_single.py --config configs/train_480/pbr.json
```

ZeRO-3 uses the matching `*_zero3.py` entrypoint and the same experiment config:

```bash
accelerate launch --config_file configs/accelerate_zero3.yaml \
  model/train_tokenlight_zero3.py --config configs/train_480/rgb.json
accelerate launch --config_file configs/accelerate_zero3.yaml \
  model/train_tokenlight_pbr_zero3.py --config configs/train_480/pbr.json
```

Replace the config filename with `rgb_decoder.json` or `pbr_decoder.json` for
decoder-space training. The latent/PBR configs use the 480 all-sets metadata and
scene cache. `rgb_decoder.json` instead uses the fixed32 dataset with
precomputed RGB VAE latents and shadow/direct masked RGB decoder losses. The PBR
pair uses depth and normal maps.

### Decoder-space loss

The opt-in decoder loss reconstructs the predicted clean latent as
`x0 = noise - velocity`, runs both prediction and target through the frozen Wan
VAE decoder, and computes the loss in decoded image space. Existing configs keep
the original latent velocity MSE.

RGB-only decoder loss:

```bash
accelerate launch --config_file configs/accelerate_zero3.yaml \
  model/train_tokenlight_zero3.py --config configs/train_480/rgb_decoder.json
```

Joint RGB + PBR decoder loss:

```bash
accelerate launch --config_file configs/accelerate_zero3.yaml \
  model/train_tokenlight_pbr_zero3.py --config configs/train_480/pbr_decoder.json
```

The relevant config keys are:

```json
{
  "tokenlight_rgb_latent_loss_weight": 0.0,
  "tokenlight_rgb_decoder_loss_weight": 1.0,
  "tokenlight_rgb_decoder_transform": "rgb",
  "tokenlight_pbr_latent_loss_weight": 0.0,
  "tokenlight_pbr_decoder_loss_weight": 1.0,
  "tokenlight_pbr_decoder_transforms": "shading:illuminance,depth:illuminance,normal:rgb",
  "tokenlight_decoder_loss_type": "mse"
}
```

`illuminance` is accepted as an alias of the Rec.709 luminance proxy used by
this repository. It is appropriate for RGB lighting/shading outputs. Keep
vector-valued normal maps in `rgb`; depth-to-luminance only treats the encoded
grayscale depth image as a scalar and is not a physical PBR renderer.
Set both latent and decoder weights above zero for a hybrid objective. Decoder
loss requires substantially more VRAM and compute because the VAE decoder stays
in the autograd graph for the prediction branch.

## TokenLight Inference

```bash
python model/infer_tokenlight.py \
  --source data/source.png \
  --attrs '{"a":0.014,"x":0.2,"y":-0.4,"z":0.8,"r":1.0,"g":1.0,"b":1.0,"lambda":1.2,"d":0.06}' \
  --checkpoint model/train/tokenlight_wan22_lora/step-100.safetensors \
  --cfg_scale 2.0 \
  --output outputs/tokenlight.png
```

## Lightoken Encoder

`model/lightoken_encoder.py` contains a standalone TokenLight-style numeric
light encoder using Gaussian Fourier features plus one projection layer per
attribute.
