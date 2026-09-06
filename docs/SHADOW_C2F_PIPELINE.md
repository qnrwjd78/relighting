# Shadow direct 480px refiner 실행 계약

> 현재 기본 모델은 별도의 60×60 coarse 출력이 없는 단일 480×480
> ResUNet입니다. 입력은 Adapter positive delta, source RGB, object mask,
> point map의 8채널이며 exact light position은 bottleneck cross-attention
> token으로 주입됩니다. 아래의 `coarse_prior_mode=adapter_delta` 명칭은
> 기존 manifest/checkpoint CLI 호환을 위해 유지되며, 실제 coarse branch를
> 의미하지 않습니다.

이 문서는 다음 파이프라인을 재현하기 위한 단일 실행 계약이다.

```text
ambient source ──> RGB baseline ──> AdapterShadow(target)
       │                                  │
       └──────────> AdapterShadow(source) ├─> coarse delta + features
                                                  │
camera geometry + exact light position ────────────┤
                                                  v
renderer GT (train/val only) ───────> Adapter-C2F refiner ─> predicted_shadow_mask
                                                            │
ambient source + predicted_shadow_mask + light tokens ──────> strict Wan RGB
```

핵심은 baseline 출력은 그림자 검출용 `target`일 뿐이고, 최종 Wan의 이미지 입력은 항상 원본 ambient `source_image`라는 점이다. Baseline RGB나 GT target RGB를 Wan source로 사용하면 안 된다.

관련 실행 파일:

- [`run_shadow_c2f_pipeline.sh`](../scripts/run_shadow_c2f_pipeline.sh): 전체 단계 orchestrator. 기본 동작은 dry-run이다.
- [`run_shadow_c2f_wan_eval.sh`](../scripts/run_shadow_c2f_wan_eval.sh): GT가 제거된 strict Wan manifest 생성, Wan 실행, mask/RGB 평가.
- [`run_shadowadapter_cache.py`](../scripts/run_shadowadapter_cache.py): 공식 AdapterShadow cache wrapper.
- [`prepare_shadow_c2f_manifest.py`](../scripts/prepare_shadow_c2f_manifest.py): 경로 정규화, scene split, GT 이름 격리.
- [`build_shadow_physics_cache.py`](../scripts/build_shadow_physics_cache.py): 기본 `--geometry-only` point-map cache. Ray-cast prior는 ablation에서만 생성.
- [`train_shadow_c2f.py`](../scripts/train_shadow_c2f.py), [`infer_shadow_c2f.py`](../scripts/infer_shadow_c2f.py): refiner 학습/추론.

## 현재 상태와 실행 금지 조건

2026-08-31 현재:

- unseen eval baseline prediction은 5,131장 생성되어 있다.
- SAM ViT-B와 EfficientNet-B1 weight는 존재한다.
- `/workspace/external/AdapterShadow/checkpoint/sbu.ckpt`는 없다. 따라서 공식 AdapterShadow 추론은 아직 시작할 수 없다.
- 8-GPU Wan shadow-mask 학습이 진행 중이다. GPU를 쓰는 AdapterShadow, VAE decode, C2F 학습/추론, Wan inference를 동시에 실행하지 않는다.
- 완성된 C2F checkpoint도 아직 없다.
- 기존 `data_train/shadow_c2f_eval_2500_2999_rgb_random3` manifest는 cache contract가 바뀌기 전에 만들어졌을 수 있다. 현재 preparer로 다시 생성해 `adapter_target_cache`, `adapter_source_cache`, `adapter_delta_cache` 세 필드가 있는지 확인한다.

두 shell script 모두 기본이 `--dry-run`이다. 현재는 `--execute`를 붙이지 않는다.

```bash
cd /workspace
bash scripts/run_shadow_c2f_pipeline.sh --stage all --dry-run
```

## 절대 지켜야 하는 데이터 경계

### 1. Wan 입력은 원본 ambient source

Strict Wan 호출은 반드시 아래 옵션을 포함한다.

```bash
--source-key source_image
```

`source_image`는 원본 manifest의 `input_image`, 즉 각 scene의 `source.png`와 같은 파일이어야 한다. Wrapper는 다음을 실행 전에 검증한다.

- `source_image`가 존재한다.
- 해석 가능한 `input_image`와 동일하다.
- `baseline_image`, `target_image`, `video`와 동일한 경로가 아니다.
- Wan-only manifest에는 `source_image`만 이미지 source로 남긴다.

### 2. GT shadow mask는 Wan manifest에 존재하지 않는다

파일 역할을 명확히 분리한다.

| 파일 | GT 포함 | 용도 |
|---|---:|---|
| prepared train/val manifest | 예: `gt_shadow_mask` | C2F supervised 학습/검증 전용 |
| C2F evaluation manifest | 예: `gt_shadow_mask` | mask/RGB 평가 전용; Wan에 직접 전달 금지 |
| C2F Wan manifest | 아니오 | C2F가 직접 생성하는 GT 제거 결과 |
| `wan_strict_no_gt.jsonl` | 아니오 | Wan inference 전용 |

Preparer는 원본의 `shadow_mask`를 `gt_shadow_mask`로 이름을 바꾸고 `shadow_mask`, `shadow_mask_pad16`를 제거한다. 이후 Wan wrapper가 allowlist 방식으로 새 manifest를 만들며 다음 항목을 모두 제거한다.

- `gt_shadow_mask`
- `shadow_mask`, `shadow_mask_pad16`
- `inf_mask`
- `target_image`, `video`
- geometry, AdapterShadow, evaluation-only 경로

Wan-only manifest에서 허용되는 shadow mask 이름은 오직 `predicted_shadow_mask`이다. 그 경로가 renderer GT와 같으면 wrapper가 실패한다.

현재 C2F inference loader는 `require_target=False`로 실행되므로 unlabeled deployment manifest에서도 동작한다. Labeled eval에서는 스크립트가 GT 없는 Wan manifest와 GT를 보존한 evaluation manifest를 동시에 분리 생성한다. Wan wrapper는 C2F Wan manifest를 그대로 신뢰하지 않고 evaluation manifest에서 allowlist manifest를 한 번 더 만들어 source/mask 계약까지 검증한다.

### 3. Mask fallback은 항상 비활성화

Strict Wan invocation은 아래 계약으로 고정한다.

```bash
--mask-key predicted_shadow_mask \
--mask-fallback-key '' \
--tokenlight_mask_tokens \
--use-mask-input \
--require-mask-input
```

`--require-mask-input` preflight는 모델을 불러오기 전에 전체 row에 대해 다음을 검사한다.

- mask 파일 존재
- 정확히 480×480
- 값이 0 또는 255뿐인 hard binary
- 흰색 255가 shadow foreground
- 빈 all-zero mask는 허용

Fallback을 비우지 않으면 기본 `mask` object mask가 conditioning으로 선택될 수 있으므로 반드시 빈 문자열을 전달한다.

## 기본 eval 경로

| 역할 | 기본 경로 |
|---|---|
| raw eval manifest | `data_train/objaverse_245_eval_2500_2999_rgb_random3/metadata_random3_heights.jsonl` |
| eval RGB/geometry root | `data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested` |
| baseline predictions | `outputs/infer/exp_1/objaverse245_eval_2500_2999_rgb_random3/rgb_baseline_epoch10/predictions` |
| prepared eval | `data_train/shadow_c2f_eval_2500_2999_rgb_random3` |
| AdapterShadow cache | `outputs/shadow_c2f/eval_2500_2999_rgb_random3/adaptershadow` |
| geometry cache | `outputs/shadow_c2f/eval_2500_2999_rgb_random3/geometry` |
| C2F inference | `outputs/shadow_c2f/eval_2500_2999_rgb_random3/c2f_infer` |
| strict Wan/eval | `outputs/shadow_c2f/eval_2500_2999_rgb_random3/wan_refined` |

`--limit N`을 사용하면 별도 `_limitN` manifest/output root를 사용하므로 full manifest를 덮어쓰지 않는다.

## 단계별 dry-run

먼저 shell 자체를 검사한다.

```bash
cd /workspace
bash -n scripts/run_shadow_c2f_pipeline.sh
bash -n scripts/run_shadow_c2f_wan_eval.sh
```

Eval branch의 각 명령을 확인한다.

```bash
bash scripts/run_shadow_c2f_pipeline.sh --stage baseline-eval --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage prepare-eval --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage adapter-eval --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage geometry-eval --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage c2f-infer --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage wan-eval --dry-run
```

GPU가 비어 있고 모든 checkpoint가 준비된 뒤에만 `--execute`로 바꾼다. 첫 검증은 반드시 작은 별도 subset으로 한다.

```bash
# 현재 고정한 조합: baseline epoch-10 + shadow-mask Wan epoch-9.
WAN_CHECKPOINT=/workspace/outputs/train/exp_1/rgb_shadow_mask_vae_7x7x5_power06_rgb_scene64_15ep_b5x8_ga1_gb40_20260830_163108/epoch-9.safetensors \
C2F_CHECKPOINT=/workspace/outputs/train/shadow_c2f/moge_adapter_subset/best.pt \
bash scripts/run_shadow_c2f_pipeline.sh \
  --stage baseline-eval,prepare-eval,adapter-eval,geometry-eval,c2f-infer,wan-eval \
  --limit 4 \
  --execute
```

## Eval branch 세부 계약

### Baseline

Baseline은 mask token 없이 ambient source와 target light 좌표로 생성한다.

```bash
python scripts/infer_manifest.py \
  --manifest data_train/objaverse_245_eval_2500_2999_rgb_random3/metadata_random3_heights.jsonl \
  --base-path data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested \
  --output-dir outputs/infer/exp_1/objaverse245_eval_2500_2999_rgb_random3/rgb_baseline_epoch10/predictions \
  --checkpoint outputs/train/exp_1/rgb_baseline_7x7x5_power06_rgb_scene64_fresh_15ep_b8_ga5_gb40_20260824_095204/epoch-10.safetensors \
  --source-key input_image \
  --height 480 --width 480 --num_frames 1 \
  --num_inference_steps 50 --cfg_scale 2.0 \
  --seed-key inference_seed --tokenlight_max_lights 2 \
  --no-tokenlight_mask_tokens --no-use-mask-input --no-with-gt --skip-existing
```

### Prepared manifest

```bash
python scripts/prepare_shadow_c2f_manifest.py \
  --input data_train/objaverse_245_eval_2500_2999_rgb_random3/metadata_random3_heights.jsonl \
  --output-dir data_train/shadow_c2f_eval_2500_2999_rgb_random3 \
  --base-path data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested \
  --baseline-dir outputs/infer/exp_1/objaverse245_eval_2500_2999_rgb_random3/rgb_baseline_epoch10/predictions \
  --adapter-cache-root outputs/shadow_c2f/eval_2500_2999_rgb_random3/adaptershadow \
  --refined-mask-root outputs/shadow_c2f/eval_2500_2999_rgb_random3/c2f_infer/refined_binary \
  --split-name eval --require-assets
```

### AdapterShadow cache

공식 공개 inference 설정을 고정한다.

- backbone `b1`
- `plug_image_adapter`, `all`, `freeze_backbone`
- `gen_pt`, grid 16×16
- raw coarse logit `> 0.9`로 prompt label 생성
- 공식 코드의 `(h,w)` prompt 좌표 순서 유지
- 입력 RGB를 1024×1024로 resize, 범위 `[0,1]`, 별도 SAM normalization 없음

```bash
python scripts/run_shadowadapter_cache.py \
  --manifest data_train/shadow_c2f_eval_2500_2999_rgb_random3/eval.jsonl \
  --source-key source_image --baseline-key baseline_image \
  --output-dir outputs/shadow_c2f/eval_2500_2999_rgb_random3/adaptershadow \
  --backend official \
  --adaptershadow-root external/AdapterShadow \
  --checkpoint external/AdapterShadow/checkpoint/sbu.ckpt \
  --sam-checkpoint external/AdapterShadow/checkpoint/sam/sam_vit_b_01ec64.pth \
  --efficientnet-checkpoint external/AdapterShadow/efficientnet/hub/checkpoints/tf_efficientnet_b1_ap-44ef0a3d.pth \
  --device cuda:0 --batch-size 1 --skip-existing
```

Cache는 480×480 NPZ로 분리된다.

| cache | 필드 |
|---|---|
| target | `final_logit`, `final_prob`, `coarse_logit`, `coarse_prob` |
| source | `final_logit`, `final_prob`, `coarse_logit`, `coarse_prob` |
| delta | `positive_delta_final_prob`, `positive_delta_coarse_prob` |

Delta는 `max(target_prob - source_prob, 0)`이다. Source cache는 scene ambient image별 content-addressed 파일이라 light마다 중복 추론/저장하지 않는다. 기본 dtype은 디스크를 고려해 float16이며 `--cache-dtype float32`로 바꿀 수 있다. Backend/weight/dtype fingerprint가 다르면 기존 cache를 재사용하지 않는다.

AdapterShadow는 조명 좌표를 입력받지 않는다. 조명별 정보는 baseline target appearance를 통해서만 들어오므로, 기본 coarse candidate는 `max(target-source, 0)`를 쓴다. C2F가 point map과 우리 manifest의 정확한 point-light position을 별도로 사용해 이 candidate의 위치와 경계를 보정한다.

### Geometry cache

기본 경로는 ray-cast mask를 생성하지 않고 MoGe point feature만 cache한다.

```bash
python scripts/build_shadow_physics_cache.py \
  --manifest data_train/shadow_c2f_eval_2500_2999_rgb_random3/eval.jsonl \
  --output-root outputs/shadow_c2f/eval_2500_2999_rgb_random3/geometry \
  --output-manifest data_train/shadow_c2f_eval_2500_2999_rgb_random3/eval_geometry.jsonl \
  --geometry-mode moge --geometry-only --device cpu --skip-existing
```

`auto`는 oracle PBR geometry가 있으면 oracle을 우선하므로 deployable 결과의 기본값으로 쓰지 않는다. Oracle은 별도 upper-bound 실험으로만 만든다.

좌표 처리는 canonical light position을 OpenCV camera frame으로 변환한 뒤 geometry cache와 같은 scale로 정규화한다. 모델은 `light_position - point_map` 상대 벡터를 coarse/fine 두 stage에 조건으로 넣는다. `--geometry-only`을 빼면 논문식 ray-cast prior ablation을 별도 root에 만들 수 있다.

### C2F inference

```bash
python scripts/infer_shadow_c2f.py \
  --manifest data_train/shadow_c2f_eval_2500_2999_rgb_random3/eval_geometry.jsonl \
  --checkpoint outputs/train/shadow_c2f/moge_adapter_subset/best.pt \
  --output-root outputs/shadow_c2f/eval_2500_2999_rgb_random3/c2f_infer \
  --output-manifest outputs/shadow_c2f/eval_2500_2999_rgb_random3/c2f_infer/wan_manifest.jsonl \
  --evaluation-manifest outputs/shadow_c2f/eval_2500_2999_rgb_random3/c2f_infer/evaluation_manifest.jsonl \
  --adapter-mode required --coarse-prior-mode adapter_delta --fallback none \
  --device cuda --batch-size 4 --skip-existing
```

주요 결과:

- `refined_probability/<scene>/<sample>.npy`: float16 probability
- `refined_probability_png/...png`: 시각화용 grayscale
- `refined_binary/...png`: strict Wan용 0/255 mask
- `wan_manifest.jsonl`: `gt_shadow_mask`가 제거된 Wan 후보 manifest
- `evaluation_manifest.jsonl`: GT를 보존한 평가 전용 manifest. Wan에 직접 전달하지 않는다.

## C2F 학습 branch

Unseen 2500–2999 eval scene로 C2F를 학습하면 안 된다. 기본 training branch는 scene 0000–1999에서 scene 단위 deterministic 80/10/10 split을 만든다.

전체 326,803개 row를 바로 처리하지 않고 기본적으로 scene 256개, scene당 16개 light를 높이 균형 선택한다. 필요하면 다음 환경변수로 바꾼다.

```bash
export TRAIN_MAX_SCENES=256
export TRAIN_MAX_PER_SCENE=16
```

단계:

```bash
bash scripts/run_shadow_c2f_pipeline.sh --stage prepare-train --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage decode-train --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage baseline-train --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage adapter-train --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage geometry-train --dry-run
bash scripts/run_shadow_c2f_pipeline.sh --stage train --dry-run
```

Training PNG가 보존되지 않았기 때문에 `decode-train`은 기존 RGB Wan VAE scene cache에서 scene별 ambient source만 복원한다. Baseline target은 같은 scene cache의 source latent와 정확한 light token으로 생성한다.

Training geometry는 PBR camera metadata가 없는 mask bundle에서도 동작하도록 기존 MoGe point map을 사용한다.

```text
data/objaverse_245_train_point_map_0000_1999/
  objaverse_fixed_7x7x5_power06_source/<scene>/source.npy
```

C2F training은 renderer `gt_shadow_mask`를 receiver 영역으로 제한하고 object 영역을 제외한 supervised target으로 사용한다. Adapter feature와 coarse prior에 dropout, morphology, noise, false-blob augmentation을 함께 적용해 detector 오류에 과적합하는 것을 줄인다.

단일 GPU 기본:

```bash
C2F_NPROC=1 C2F_BATCH_SIZE=4 \
bash scripts/run_shadow_c2f_pipeline.sh --stage train --dry-run
```

향후 여러 GPU가 비었을 때:

```bash
C2F_NPROC=8 C2F_BATCH_SIZE=4 \
bash scripts/run_shadow_c2f_pipeline.sh --stage train --execute
```

현재 8-GPU Wan 학습이 끝나기 전에는 위 명령을 실행하지 않는다.

Checkpoint는 `latest.pt`, `epoch-NNN.pt`, `best.pt`로 저장되며 schema는 `tokenlight_shadow_c2f_checkpoint_v1`이다.

## Strict Wan + 평가

Standalone dry-run:

```bash
bash scripts/run_shadow_c2f_wan_eval.sh \
  --manifest outputs/shadow_c2f/eval_2500_2999_rgb_random3/c2f_infer/evaluation_manifest.jsonl \
  --checkpoint /path/to/finished-shadow-mask-wan.safetensors \
  --baseline-dir outputs/infer/exp_1/objaverse245_eval_2500_2999_rgb_random3/rgb_baseline_epoch10/predictions \
  --output-root outputs/shadow_c2f/eval_2500_2999_rgb_random3/wan_refined \
  --base-path data/objaverse_245_eval_2500_2999/unseen_7x7x5_power06_png_exclude_requested \
  --dry-run
```

실행 시 wrapper가 먼저 `manifests/wan_strict_no_gt.jsonl`을 atomic write한다. 이후 Wan은 다음을 명시적으로 사용한다.

```text
source       = source_image             # ambient source
mask         = predicted_shadow_mask    # C2F binary result
mask fallback= disabled
target       = disabled
with-GT panel= disabled
```

평가는 원래 `evaluation_manifest.jsonl`로만 수행한다.

Mask methods:

- AdapterShadow target final probability
- AdapterShadow positive target-source delta
- refined continuous probability
- refined binary mask

Metrics는 IoU, Dice, precision/recall, specificity, BER, boundary F, centroid offset, principal-axis angle error, area ratio를 전체 및 light height별로 기록한다. Test/eval manifest에서 `--calibrate`를 사용하지 않는다. Threshold calibration이 필요하면 val split에서 정한 값을 고정해 eval에 전달한다.

RGB 평가는 baseline과 refined-mask Wan 출력을 다음 영역별로 비교한다.

- full image
- object
- receiver
- shadow, shadow core, shadow boundary
- lit receiver

LPIPS는 `--lpips`를 명시할 때만 활성화한다.

## 주요 환경변수

| 변수 | 의미 |
|---|---|
| `PYTHON_BIN` | 일반 pipeline Python |
| `ADAPTER_PYTHON_BIN` | 공식 AdapterShadow dependency가 설치된 Python |
| `ADAPTER_CHECKPOINT` | official `sbu.ckpt` |
| `ADAPTER_DEVICE`, `ADAPTER_BATCH_SIZE` | 기본 `cuda:0`, 1 |
| `GEOMETRY_DEVICE` | 기본 `cpu` |
| `GEOMETRY_MODE` | eval 기본 `moge` |
| `C2F_CHECKPOINT` | C2F `best.pt` |
| `C2F_NPROC`, `C2F_BATCH_SIZE` | C2F DDP process 수와 per-process batch |
| `WAN_CHECKPOINT` | mask-conditioned Wan checkpoint (기본값: 해당 run의 epoch-9) |
| `WAN_GPU_DEVICES` | Wan multi-process inference GPU 목록 |
| `WAN_EVAL_LPIPS=1` | regional LPIPS 활성화 |

모든 경로 변수와 기본값은 [`run_shadow_c2f_pipeline.sh`](../scripts/run_shadow_c2f_pipeline.sh)에 기록되어 있다.

## Resume와 stale cache 주의

- AdapterShadow NPZ는 atomic write이며 schema/backend/weight/dtype fingerprint를 검사한다. `--skip-existing`가 안전하다.
- Source Adapter cache는 scene별 공유된다.
- Geometry cache는 point map, object/receiver mask, geometry mode, feature scale signature를 cache ID에 포함한다. 논문식 physics prior는 별도 ablation으로만 유지한다.
- C2F inference는 checkpoint SHA-256, model config, threshold, morphology/fallback 설정, 입력 cache ID를 sample cache fingerprint에 포함한다. `--skip-existing`는 metadata와 artifact signature가 모두 맞을 때만 재사용한다.
- `float16` cache는 storage 절약을 위한 기본값이다. 분석용 raw logit 정밀도가 중요하면 AdapterShadow만 `--cache-dtype float32`로 새 root에 생성한다.

## 완료 전 체크리스트

- [ ] 현재 8-GPU Wan 학습 종료 및 GPU idle 확인
- [ ] `sbu.ckpt` 확보 및 official asset preflight 통과
- [ ] train/val/test scene ID 교집합이 0인지 확인
- [ ] prepared manifest에 세 Adapter cache field 존재
- [ ] geometry eval은 `moge`, oracle/physics ablation은 별도 root인지 확인
- [ ] C2F `best.pt`의 validation metric 확인
- [ ] `wan_strict_no_gt.summary.json`에서 `contains_gt_shadow_mask=false`
- [ ] Wan resolved config가 `source_key=source_image`인지 확인
- [ ] Wan resolved config가 `mask_key=predicted_shadow_mask`, fallback empty인지 확인
- [ ] required-mask preflight의 fallback row가 0인지 확인
- [ ] mask/RGB metric 모두 expected row 수와 evaluated row 수가 같은지 확인
