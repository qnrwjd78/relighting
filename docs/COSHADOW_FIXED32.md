# Fixed32 CoShadow / MultiShadow adaptation

This is an isolated single-object adaptation of MultiShadow (formerly
CoShadow), arXiv:2603.02743. It does not modify the successful TokenLight base
trainer or `model/tokenlight_wan.py`.

## What is preserved, and what differs

The paper conditions Stable Diffusion 1.5 on a shadow-free composite and an
object mask, predicts each shadow box with a separate frozen network, quantizes
XYXY to a 16x16 grid, and injects four learned positional tokens through the
text/cross-attention pathway.

Fixed32 has one source image reused for up to 32 target lights. A predictor that
sees only source+object mask therefore has multiple correct boxes for identical
inputs. This implementation intentionally adds the existing `attrs_json`
Lightoken condition to both the box predictor and diffusion model. The box
predictor uses RGB, object mask, normalized XY CoordConv channels, and Lightoken
features.

Wan uses a frozen T5 text stack and has no safe vocabulary-extension contract
in this repository. The four learned `[sx1,sy1,sx2,sy2]` embeddings are thus
inserted as clean DiT self-attention prefix tokens, beside source, object-mask,
and light tokens. This is a Wan adaptation of the paper's spatial grounding,
not a claim of architectural identity. Prefix tokens receive timestep zero;
only target tokens receive the sampled flow timestep.

The paper's attention-alignment loss requires exact token-specific
cross-attention maps from selected middle/late layers. The current Wan blocks do
not expose those maps. No proxy is presented as equivalent:
`coshadow_attention_alignment_weight != 0` fails immediately. The implemented
objectives are flow-matching MSE and a latent-grid shadow-mask BCE head. A
current-flow-state L1 background-preservation term exists as an opt-in
ablation, but its default weight is zero: unlike the paper's shadow-free
composite, fixed32 `source.png` is ambient/HDRI while the target uses a new
point light, so valid receiver/object illumination changes occur outside the
shadow mask and must not be suppressed by default.

## 1. Build scene-disjoint metadata

The builder fully decodes target/source/object/shadow PNGs, requires matching
480x480 sizes, validates all eight fixed32 light values, preserves every input
field, supports empty shadow masks, and uses a stable SHA-256 scene split. The
canonical fixed32 reject list is applied by default, removing complete scenes.

```bash
python scripts/build_fixed32_coshadow_metadata.py --dry-run --limit 128

python scripts/build_fixed32_coshadow_metadata.py \
  --output-dir data_train/objaverse_fixed32_coshadow_480
```

Output fields use normalized half-open `xyxy`: `x2/y2` are one pixel past the
last foreground pixel. Empty masks store `[0,0,0,0]`, bins `[0,0,0,0]`, and
`coshadow_bbox_valid=false`. Non-empty coordinates are quantized by nearest bin
with `round(coord * 15)`.

Use `--include-rejected` only for an explicit population-changing ablation. It
is not comparable with the other fixed32 methods trained after applying
`data/objaverse_fixed32_png/reject_metadata.txt`.

## 2. Validate and train the box predictor

Validation-only mode performs no model construction or weight loading:

```bash
python model/train_coshadow_box_predictor.py \
  --config configs/train_480/coshadow_box.json \
  --validate_dataset_only --validate_dataset_samples 256
```

Single GPU:

```bash
python model/train_coshadow_box_predictor.py \
  --config configs/train_480/coshadow_box.json
```

ZeRO-3:

```bash
accelerate launch --config_file configs/accelerate_zero3.yaml \
  model/train_coshadow_box_predictor.py --train_mode zero3 \
  --config configs/train_480/coshadow_box.json
```

The loss is `L1 + lambda_iou * (1-IoU) + lambda_presence * BCE`. L1 and IoU
are evaluated only for valid boxes; BCE learns the empty-shadow case. Single
GPU keeps FP32 AdamW parameters and uses BF16 autocast.

## 3. Train the diffusion adaptation

The primary config follows the paper's train/inference alignment and consumes
the frozen predicted boxes from `outputs/train/coshadow_box_fixed32/final`:

Before training, validate the diffusion metadata, full PNG decodes, bbox/bin
contract, light values, and a sample of cached latents without constructing Wan
or loading the box predictor:

```bash
python model/train_tokenlight_coshadow.py \
  --config configs/train_480/coshadow_diffusion.json \
  --validate_dataset_only --validate_dataset_samples 256
```

The checkpoint path may be absent during this validation-only pass; it is
required once actual `bbox_source=predicted` training starts.

```bash
python model/train_tokenlight_coshadow.py \
  --config configs/train_480/coshadow_diffusion.json
```

ZeRO-3:

```bash
accelerate launch --config_file configs/accelerate_zero3.yaml \
  model/train_tokenlight_coshadow.py --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/coshadow_diffusion.json
```

For an explicitly labelled oracle/debug ablation only, override
`--coshadow_bbox_source gt --coshadow_box_predictor_checkpoint none`. GT boxes
must not be used as the primary result because they introduce a training and
deployment mismatch relative to the paper.

The RGB target/source cache is reused. Object masks may be encoded on the fly,
or a compatible object-mask latent cache can be supplied through
`coshadow_object_mask_latent_cache_dir`. Shadow-mask PNGs remain raw targets for
the auxiliary loss. The default per-GPU batch is one because source, object
mask, layout, and light prefixes increase sequence length.

## Checkpoint contract

Box checkpoints are directories containing `model.safetensors` and
`config.json`; the config records `use_coordconv` and model dimensions.
Diffusion checkpoints export LoRA, the original light/type embeddings, the four
layout-token module, and the optional mask head. The layout bin count and token
order must remain fixed between training and inference.
