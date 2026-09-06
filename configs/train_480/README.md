# Canonical 480 training configs

The latent RGB/PBR and PBR-decoder configs use the all-sets scene cache:

- Metadata: `data_train/tokenlight_all_3sets/metadata.jsonl`
- Scene latent cache: `data/vae_cache_480`
- Resolution: 480 x 480
- PBR maps: depth + normal
- Micro-batch: 1
- Gradient accumulation: 24

| Config | Trainer | Loss space | Outputs supervised |
| --- | --- | --- | --- |
| `rgb.json` | RGB | latent velocity | RGB |
| `pbr.json` | PBR | latent velocity | RGB, depth, normal |
| `rgb_decoder.json` | RGB | decoded pixels | RGB |
| `pbr_decoder.json` | PBR | decoded pixels | RGB, depth, normal |

Decoder-space configs use the original metadata PNGs as ground truth. Only the
prediction is VAE-decoded; cached target latents are still used to construct the
diffusion/noise target, but they are not decoded for the pixel-space loss.

`rgb_decoder.json` uses `data/objaverse_fixed32_png` and
`data_train/objaverse_fixed32_480/metadata.jsonl`. It decodes RGB and optimizes

```text
1.0 * full-image loss + 0.1 * shadow-mask loss + 0.1 * direct-lit-mask loss
```

The mask metadata keys are `shadow_mask` and `inf_mask`. RGB target/source VAE
latents are read from `data/vae_cache_fixed32_480/rgb`; masks remain ordinary
PNG files because they are applied after decoding.

Build the fixed32 cache on four GPUs before training:

```bash
python utils/build_latent_cache.py \
  --mode rgb \
  --data-root data/objaverse_fixed32_png \
  --metadata-path data_train/objaverse_fixed32_480/metadata.jsonl \
  --output-dir data/vae_cache_fixed32_480/rgb \
  --height 480 --width 480 \
  --gpu-devices 0,1,2,3 \
  --batch-size 8 --num-workers 4 --shard-size 512
```

The builder deduplicates repeated paths, partitions unique PNGs across GPUs,
writes safetensors shards, and merges the per-GPU indexes into `index.jsonl`.

`rgb_decoder.json` also conditions the DiT on the `shadow_mask` metadata image.
Build that mask-token cache separately so the RGB target/source cache does not
need to be rebuilt:

```bash
python utils/build_latent_cache.py \
  --mode custom --image-keys shadow_mask --transform none \
  --data-root data/objaverse_fixed32_png \
  --metadata-path data_train/objaverse_fixed32_480/metadata.jsonl \
  --output-dir data/vae_cache_fixed32_480/shadow_mask \
  --height 480 --width 480 \
  --gpu-devices 0,1,2 \
  --batch-size 13 --num-workers 4 --shard-size 512
```

The cached shadow-mask latent is used as a model input token. The original
shadow-mask PNG is independently loaded as the region mask for decoder loss.

## Launch

Use the same experiment JSON for single GPU and ZeRO-3. Only the entrypoint and
launcher change.

```bash
# Single GPU RGB example
python model/train_tokenlight_single.py --config configs/train_480/rgb.json

# ZeRO-3 RGB example
accelerate launch --config_file configs/accelerate_zero3.yaml \
  model/train_tokenlight_zero3.py --config configs/train_480/rgb.json

# Single GPU PBR example
python model/train_tokenlight_pbr_single.py --config configs/train_480/pbr.json

# ZeRO-3 PBR example
accelerate launch --config_file configs/accelerate_zero3.yaml \
  model/train_tokenlight_pbr_zero3.py --config configs/train_480/pbr.json
```

Use the corresponding `_decoder.json` filename for decoder-space loss.

## Changing maps or data

Changing PBR maps requires coordinated edits to:

- `dataset.data_file_keys`
- `pbr.tokenlight_pbr_streams`
- `pbr.tokenlight_pbr_stream_image_keys`
- `pbr.tokenlight_pbr_stream_loss_weights`
- `loss.tokenlight_pbr_decoder_transforms` for decoder training

For example, albedo + roughness needs metadata keys `pbr_albedo_image` and
`pbr_roughness_image` and matching cached latents. Changing datasets requires
updating `dataset_base_path`, `dataset_metadata_path`, and
`scene_latent_cache_root`; the metadata must expose every configured image key.
