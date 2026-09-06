# 물리 조건부 relighting 학습 4종 설계와 실행법

이 문서는 `objaverse_fixed32_png`를 공통 데이터로 사용하는 다음 네 가지 실험을 정리한다.

1. UniRelight에서 착안한 RGB + depth + normal 공동 생성(PBR)
2. CoShadow로 부르던 2026년 논문의 현재 버전인 MultiShadow 기반 shadow-layout 조건부 생성
3. LGI(Light-Geometry Interaction) map 조건부 생성
4. 정답 direct-lit/cast-shadow mask를 넣는 oracle 진단 실험(object mask는 검증 전용)

중요한 전제는 이 네 구현이 논문 코드를 그대로 옮긴 **exact reproduction이 아니라**, 현재 Wan 2.2 5B + TokenLight + rectified-flow 학습 체계에 맞춘 연구용 adaptation이라는 점이다. 논문과 동일한 부분과 바꾼 부분을 아래에서 분리해 적는다. 기존 latent-only 기준 구현인 `model/train.py`는 건드리지 않고 별도 entrypoint를 사용한다.

네 신규 trainer가 공유하는 안전 실행 루프도 새 파일
`model/tokenlight_physics_runtime.py`에 분리했다. Single GPU에서는 FP32
trainable parameter + BF16 autocast를 사용하고, ZeRO-3 precision은
DeepSpeed가 관리한다. 학습률 scheduler는 첫 step부터 실제 상수인 no-op
`LambdaLR`이며 기존 launcher 파일은 수정하지 않는다.

공식 자료:

- [UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting](https://papers.nips.cc/paper_files/paper/2025/hash/9bb159d886a92a8c3375dc6cbb383333-Abstract-Conference.html), [NVIDIA 프로젝트 페이지](https://research.nvidia.com/labs/toronto-ai/UniRelight/)
- [MultiShadow: Multi-Object Shadow Generation for Image Compositing via Diffusion Model](https://arxiv.org/abs/2603.02743) — 초기 작업명/요청명은 CoShadow이고, 현재 arXiv 제목은 MultiShadow이다.
- [Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps](https://arxiv.org/abs/2602.21820)

## 구현 요약

| 실험 | 학습 entrypoint | 추가 입력 또는 공동 출력 | 현재 목적 |
|---|---|---|---|
| PBR | `model/train_tokenlight_pbr_safe.py` | depth, normal을 RGB와 함께 조건/target stream으로 사용 | intrinsic 공동 예측이 RGB relighting을 안정화하는지 확인 |
| MultiShadow adaptation | `model/train_coshadow_box_predictor.py` 후 `model/train_tokenlight_coshadow.py` | object mask, light-conditioned shadow box, shadow mask supervision | 명시적인 shadow 위치 grounding 효과 확인 |
| LGI adaptation | `model/train_tokenlight_lgi.py` | `[min, max, nearest-to-zero, validity]` dense map | light와 geometry가 결합된 occlusion prior 효과 확인 |
| GT-mask oracle | `model/train_tokenlight_gt_masks.py` | `[direct-lit, cast-shadow]` 2채널 정답 dense map; object는 관계 검증 전용 | 모델이 완전한 공간 힌트를 받으면 문제를 풀 수 있는지 상한선 확인 |

네 경로 모두 현재 기본 설정에서는 RGB decoder loss를 사용하지 않는다. RGB target에는 기존 Wan/TokenLight rectified-flow velocity loss를 사용한다. 따라서 “한 random timestep의 예측을 바로 decode해서 image loss를 거는” 이전 문제와 분리되어 있고, 매 학습 step에서 전체 denoising trajectory를 unroll하지 않는다.

## 공통 데이터 계약

기준 파일은 다음과 같다.

- 원본 root: `data/objaverse_fixed32_png`
- 메타데이터: `data_train/objaverse_fixed32_480/metadata.jsonl`
- 제외 scene 목록: `data/objaverse_fixed32_png/reject_metadata.txt`
- RGB VAE cache: `data/vae_cache_fixed32_480/rgb`

현재 점검 기준 원본 metadata는 48,835행, 1,681 scene이며 reject 목록의 523 scene을 제외하면 35,614행이다. 모든 비교 실험에서 같은 reject 목록을 적용해야 한다.

한 행의 핵심 key는 다음 의미를 갖는다.

| key | 의미 |
|---|---|
| `video` | target-light RGB 정답 `position_NNN.png` |
| `input_image` | scene별 source RGB `source.png` |
| `attrs_json` | point-light 위치, 색, 세기/크기 등 Lightoken 조건 |
| `mask` | object silhouette |
| `inf_mask` | target light에서 geometry ray가 직접 도달하는 object 내부 영역 |
| `shadow_mask` | object가 receiver에 만든 cast-shadow 영역; object 내부는 제외 |
| `pbr_depth_image` | scene depth PNG |
| `pbr_normal_image` | scene normal PNG |

모든 dense map은 target과 같은 카메라, crop, flip, 480×480 좌표계를 유지해야 한다. RGB와 map에 서로 다른 crop/flip augmentation을 적용하면 물리 조건이 오히려 잘못된 supervision이 된다.

현재 config는 source/target RGB latent cache를 사용하지만 PBR depth/normal은 PNG를 읽어 VAE로 실시간 encode한다. 기능적으로는 맞지만 처리량이 느릴 수 있으므로 장기 학습 전에는 depth/normal 전용 cache가 필요할 수 있다.

## 1. UniRelight 기반 PBR: albedo 대신 depth + normal

### 논문에서 가져온 핵심

UniRelight는 source, relit RGB, albedo latent를 한 DiT sequence에 넣고 type embedding으로 각 modality를 구분해 relighting과 intrinsic decomposition을 공동 학습한다. NVIDIA 설명에서도 relit video와 albedo를 한 번에 공동 예측하는 것이 핵심이다.

논문의 synthetic-data conditioning 비율은 다음과 같다.

- 70%: source를 조건으로 주고 relit RGB와 intrinsic을 공동 예측
- 18%: source와 정답 intrinsic을 조건으로 주고 RGB를 예측
- 12%: source를 버리고 정답 intrinsic만 조건으로 주고 RGB를 예측

논문의 albedo auxiliary weight는 0.1이다.

### 이 저장소에서 바꾼 부분

- albedo 한 stream을 `depth`, `normal` 두 stream으로 교체했다.
- albedo weight 0.1을 기준으로 시작했으나, 현재는 `depth=0.1`, `normal=0.1`로 설정해 auxiliary 총 weight가 0.2다.
- environment-map feature 대신 기존 `attrs_json` Lightoken을 사용한다.
- video가 아니라 1-frame 480×480 image 학습이다.
- UniRelight의 원 backbone 대신 Wan 2.2 TI2V 5B와 LoRA를 사용한다.
- 모든 auxiliary objective는 VAE latent velocity MSE이며 decoder/image loss는 금지한다.

한 batch sample은 하나의 mode만 고른다. 기본 `0.70/0.18/0.12` 설정에서 depth와 normal은 함께 noisy target이 되거나 함께 clean condition이 된다. source-drop mode에서는 source latent를 0으로 만들되 RGB latent loss는 유지한다.

개념적으로 현재 loss는 다음과 같다.

```text
L = w_scheduler(t) * [
      1.0 * L_velocity(RGB)
    + 0.1 * I(PBR-target) * L_velocity(depth)
    + 0.1 * I(PBR-target) * L_velocity(normal)
]
```

PBR condition stream은 clean latent와 `t=0` timestep modulation을 쓰고, target stream은 RGB와 같은 sigma에서 독립 noise를 섞는다. stream type embedding으로 RGB/source/depth/normal의 역할을 구분한다.

관련 파일:

- trainer: `model/train_tokenlight_pbr_safe.py`
- isolated joint-stream model function: `model/tokenlight_wan_pbr_safe.py` (`model/tokenlight_wan_pbr.py`의 기존 동작은 재사용만 한다.)
- matching inference wrapper: `scripts/infer_manifest_pbr_safe.py` (`scripts/infer_manifest_pbr.py`는 수정하지 않고 I/O만 재사용한다.)
- config: `configs/train_480/fixed32_pbr_depth_normal.json`

성공한 기존 RGB TokenLight checkpoint로 시작할 때 type embedding의 의미가 어긋나지 않도록 migration도 포함한다. 기존 4행 schema `[source=0, mask=1, light=2, target=3]`에서 `source 0 -> PBR source 0`, `light 2 -> PBR light 1`, `target 3 -> RGB target 2와 각 PBR target 행`으로 복사한다. depth/normal condition 행은 기존 RGB schema에 대응 의미가 없으므로 새 초기값을 유지한다. 이미 PBR schema와 shape가 맞는 checkpoint는 그대로 복원한다.

### 데이터 확인과 실행

아래 명령은 컨테이너 내부 `/workspace`를 현재 디렉터리로 가정한다.

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_pbr_safe.py \
  --train_mode single \
  --config configs/train_480/fixed32_pbr_depth_normal.json \
  --validate_dataset_only \
  --dataset_validation_samples 32
```

single GPU 학습:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_pbr_safe.py \
  --train_mode single \
  --config configs/train_480/fixed32_pbr_depth_normal.json
```

ZeRO-3 학습에서 `N`은 사용할 GPU 수이다.

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_tokenlight_pbr_safe.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/fixed32_pbr_depth_normal.json
```

현재 시작 설정은 RGB weight 1.0, depth/normal 각 0.1, batch 1, gradient accumulation 40이다. depth와 normal PNG의 단위와 normal 좌표계는 전체 dataset에서 동일해야 한다. 현재 preflight는 PNG decode와 shape를 검사하지만 depth metric scale이나 normal의 OpenGL/OpenCV 방향까지 판별하지는 않는다.

safe checkpoint inference는 학습과 같은 per-stream timestep semantics를 쓰는 별도 wrapper로 실행한다. 아래의 checkpoint 경로는 실제 run의 epoch/step 파일로 바꾼다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_manifest_pbr_safe.py \
  --manifest data_train/objaverse_fixed32_480/metadata.jsonl \
  --base-path data/objaverse_fixed32_png \
  --output-dir outputs/infer/pbr_depth_normal_safe \
  --checkpoint outputs/train/tokenlight_480_fixed32_pbr_depth_normal_safe_TIMESTAMP/epoch-X.safetensors \
  --height 480 \
  --width 480 \
  --pbr-mode condition \
  --limit 32
```

`--pbr-mode condition`은 정답 depth/normal을 clean condition으로 넣어 RGB를 생성한다. RGB와 depth/normal을 함께 denoise해 공동 출력 자체를 검사하려면 `--save-pbr-predictions`를 추가한다.

## 2. MultiShadow(CoShadow) 기반 구현

### 논문에서 가져온 핵심

MultiShadow는 shadow-free composite와 object mask를 사용하는 dense image pathway, 그리고 object별 shadow bounding box를 discrete positional token으로 바꾸는 prompt pathway를 함께 쓴다. 원 논문의 주요 구성은 다음과 같다.

- 별도 shadow-box predictor를 먼저 학습한 뒤 freeze한다.
- box는 normalized `(x1, y1, x2, y2)`이고 16×16 grid로 양자화한다.
- 원 모델은 box token을 CLIP text sequence에 삽입하고 UNet cross-attention으로 주입한다.
- dense image pyramid는 GAAM을 통해 `GN(x) * (1 + Delta(L)) + B(L)` 형태로 UNet을 조절한다.
- diffusion, shadow-mask, background-preservation, attention-alignment loss를 합친다.
- attention alignment는 box token의 cross-attention map과 downsampled shadow mask 사이의 KL loss다.

논문 설정은 SD 1.5, 512×512, diffusion batch 4/lr 1e-5/80 epoch이며 box predictor는 batch 8/lr 1e-4/up to 100 epoch이다.

### fixed32에 필요한 adaptation

fixed32에서는 하나의 `source.png`가 32개 target light에 공통으로 사용된다. 따라서 source와 object mask만으로는 32개의 서로 다른 shadow box 중 무엇을 예측해야 하는지 결정할 수 없다. 로컬 box predictor는 이 모호성을 없애기 위해 반드시 `attrs_json` Lightoken도 입력받는다.

또한 fixed32 `source.png`는 논문의 “같은 target light에서 shadow만 제거한 composite”와 동일한 데이터가 아니다. 이 경로는 순수 shadow insertion이 아니라 **relighting과 shadow synthesis를 동시에 학습하는 MultiShadow-inspired adaptation**이다.

Wan에는 SD UNet의 동일한 CLIP cross-attention/skip 구조가 없으므로 신규 adaptation 모듈에서는 다음처럼 바꿨다.

- dense pathway: source latent + object-mask latent를 Wan self-attention prefix로 넣는다.
- layout pathway: `(x1,y1,x2,y2)`를 네 개의 learned x/y positional prefix token으로 넣는다.
- target-grid hidden feature에서 auxiliary shadow-mask head를 학습한다.
- target velocity MSE + weighted mask BCE를 기본으로 사용하며 shadow 밖 latent background L1은 선택적 ablation으로만 제공한다.
- exact GAAM과 CLIP prompt injection은 구현하지 않았다.
- exact attention-alignment loss는 Wan에서 대응 cross-attention map을 노출하지 않으므로 `coshadow_attention_alignment_weight=0`으로 고정하며, 0이 아니면 조용히 무시하지 않고 즉시 실패한다.

기본 diffusion loss는 다음과 같다.

```text
L = 1.0 * L_rectified_flow
  + 0.1 * L_shadow_mask_BCE(positive_weight=4)
  + 0.0 * L_background_latent_L1  # 기본 off; optional ablation
```

background-preservation loss를 기본 0으로 둔 이유는 fixed32 `source.png`가 target point light에서 shadow만 지운 영상이 아니라 ambient/HDRI source이기 때문이다. 논문처럼 source 배경과 target의 비-shadow 배경이 동일하다는 가정이 성립하지 않으므로, 이를 강제하면 올바른 target-light 변화까지 억제할 수 있다.

### 단계 0: metadata와 box target 생성

builder는 기존 target/source/object-mask/shadow-mask PNG를 끝까지 decode하고, `shadow_mask`에서 half-open normalized XYXY box를 계산한다. 빈 shadow mask는 오류가 아니라 `coshadow_bbox_valid=false`인 유효 sample이다. 16-bin 좌표와 light attrs를 보존하고, stable SHA-256 scene hash로 90/5/5 train/val/test split을 만들어 동일 scene이 split을 넘지 못하게 한다.

fixed32 adaptation에 필요한 per-light supervision은 다음과 같다.

- `video`: 최종 RGB target
- `input_image`: 현재 ambient/HDRI source
- `mask`: object silhouette
- `shadow_mask`: receiver 위 cast-shadow binary mask
- `attrs_json`: target point light
- builder가 `shadow_mask`로부터 만드는 normalized/quantized bbox와 presence

이 항목은 현재 fixed32 metadata에 이미 들어 있으므로 image를 다시 합성하지 않고 검증된 CoShadow manifest를 만든다. exact MultiShadow 데이터까지 만들려면 각 target light마다 “object appearance와 background는 target과 같고 해당 object의 cast shadow만 제거된 composite”를 별도로 render/inpaint해야 한다. 현재 `source.png`는 그 파일을 대신할 수 없으므로 그 exact 데이터 생성은 이 adaptation의 범위 밖이다.

먼저 일부만 read-only 점검한다.

```bash
python scripts/build_fixed32_coshadow_metadata.py \
  --dry-run \
  --limit 128
```

전체 metadata 생성:

```bash
python scripts/build_fixed32_coshadow_metadata.py
```

출력은 `data_train/objaverse_fixed32_coshadow_480/{train,val,test,metadata}.jsonl`과 `summary.json`이다. 기본적으로 canonical reject list를 적용한다. `--include-rejected`는 품질 문제가 확인된 scene까지 의도적으로 다시 포함하는 옵션이므로 일반 학습에는 쓰지 않는다.

생성된 diffusion manifest, PNG, bbox quantization, light attrs, RGB latent cache를 모델을 올리지 않고 함께 점검할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_coshadow.py \
  --train_mode single \
  --config configs/train_480/coshadow_diffusion.json \
  --validate_dataset_only \
  --validate_dataset_samples 128
```

`--validate_dataset_samples 0`은 모든 metadata row의 계약을 검사하고 모든 관련 PNG와 cache sample을 읽으므로 오래 걸릴 수 있다.

### 단계 1: light-conditioned shadow-box predictor

입력은 source RGB, object mask, CoordConv `(x,y)`, Lightoken attrs다. 출력은 normalized XYXY와 shadow-presence logit이며 loss는 다음과 같다.

```text
L_box = L1(valid boxes) + [1 - IoU(valid boxes)] + BCE(shadow presence)
```

dataset-only 점검:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_coshadow_box_predictor.py \
  --train_mode single \
  --config configs/train_480/coshadow_box.json \
  --validate_dataset_only \
  --validate_dataset_samples 128
```

single GPU 학습:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_coshadow_box_predictor.py \
  --train_mode single \
  --config configs/train_480/coshadow_box.json
```

ZeRO-3 학습:

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_coshadow_box_predictor.py \
  --train_mode zero3 \
  --config configs/train_480/coshadow_box.json
```

최종 checkpoint directory는 기본적으로 `outputs/train/coshadow_box_fixed32/final`이다. 내부의 `model.safetensors`와 `config.json`을 함께 유지해야 한다.

### 단계 2: Wan diffusion/rectified-flow 학습

`configs/train_480/coshadow_diffusion.json`의 기본 `coshadow_bbox_source`는 freeze한 predictor를 쓰는 `predicted`이며 checkpoint 기본 경로는 `outputs/train/coshadow_box_fixed32/final`이다. 즉 단계 1을 먼저 끝내야 기본 명령이 실행된다. `gt`는 box token 경로만 분리해 점검하는 oracle/smoke override다.

GT box smoke 학습:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_coshadow.py \
  --train_mode single \
  --config configs/train_480/coshadow_diffusion.json \
  --coshadow_bbox_source gt
```

freeze한 predicted box를 쓰는 single GPU 본 학습:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_coshadow.py \
  --train_mode single \
  --config configs/train_480/coshadow_diffusion.json
```

ZeRO-3 본 학습:

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_tokenlight_coshadow.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/coshadow_diffusion.json
```

관련 파일은 `scripts/build_fixed32_coshadow_metadata.py`, `model/coshadow_box_predictor.py`, `model/train_coshadow_box_predictor.py`, `model/tokenlight_wan_coshadow.py`, `model/train_tokenlight_coshadow.py`다.

## 3. LGI 기반 구현

### 논문의 LGI map 생성

LGI는 단순 depth 입력이 아니라 “이 light에서 이 geometry가 어느 방향으로 가려지는가”를 명시적으로 만든다. 논문 알고리즘은 다음과 같다.

1. 카메라 intrinsic `K`와 depth `D(u,v)`로 각 pixel을 카메라 좌표 3D point로 lift한다.

   ```text
   p = D(u,v) * K^-1 * [u, v, 1]^T
   ```

2. `p`에서 point light `l`로 향하는 ray 위에 `N=16`개 점을 균일하게 뽑는다.

   ```text
   S_n = p + delta_n * (l - p)
   ```

3. 각 `S_n`을 image plane으로 재투영해 해당 pixel의 depth를 읽고, 다시 surface point `S'_n`으로 만든다. camera frustum 밖이나 infinite-depth sample은 invalid로 제외한다.

4. camera plane normal `n=[0,0,1]^T`에 대한 sample surface와 light ray의 elevation 차이를 구한다.

   ```text
   v_s = S'_n - p
   v_l = l - p
   e_s = asin((v_s dot n) / ||v_s||)
   e_l = asin((v_l dot n) / ||v_l||)
   e_d[n] = e_s - e_l
   ```

5. 유효 sample 축을 줄여 세 채널을 만든다. 순서는 반드시 아래와 같다.

   ```text
   c1 = min_n e_d[n]
   c2 = max_n e_d[n]
   c3 = e_d[argmin_n |e_d[n]|]
   channel order = [min, max, nearest-to-zero]
   ```

논문 정의에서 elevation difference의 범위는 자연스럽게 `(-pi, pi)`다. 다만 현재 저장된 예시 NPY에는 단위 provenance metadata가 없으며 loader도 radian 여부를 변환하거나 추정하지 않는다. 구현은 파일의 signed scalar 세 채널을 float32로 그대로 읽고 identity normalization을 적용한다. 세 값이 모두 0인 pixel을 invalid로 표시하는 네 번째 local `validity` 채널을 붙이는데, `validity`는 논문 map이 아니라 missing/invalid ambiguity를 줄이기 위한 로컬 파생 채널이다. 새 map 생성기는 논문 식에 따라 radian 값을 저장하고 모든 scene에서 같은 convention을 지켜야 한다.

### 필요한 디렉터리 형식

```text
data/objaverse_32_lgimap/
  scenes/
    scene_000001/
      position_00/
        min.npy
        max.npy
        nearest.npy
      position_01/
        ...
```

metadata의 `position_000`은 LGI의 `position_00`, `position_031`은 `position_31`로 매핑한다. 행에 `lgi_dir`이 있으면 그 경로가 우선한다. 각 NPY는 정확히 `(480,480)`의 finite numeric array여야 하며 loader가 pixel마다 `min <= nearest <= max`를 강제한다.

현재 실제 예시는 `scene_000001/position_00` 한 개뿐이다. 이 상태에서는 해당 sample만 제한한 preflight는 통과할 수 있지만 전체 학습은 의도적으로 missing-file error를 낸다. 35,614개 사용 행에 해당하는 LGI를 모두 채운 뒤 full preflight를 통과시켜야 한다.

### 논문과 이 구현의 차이

LGI 논문 backbone은 SDXL latent bridge matching이다.

```text
z(t) = (1-t)z0 + t*z1 + sigma*sqrt(t(1-t))*epsilon
v_target = (z1-z(t))/(1-t)
z1_hat = z(t) + (1-t)*v_theta
```

논문은 latent loss와 brightness-change 영역을 강조한 decoded pixel L1을 함께 사용한다. 그 설정은 threshold 0.01, dilation kernel 17, pixel-loss weight 10, batch 5, lr 3e-5다.

현재 구현은 이 bridge와 decoder loss를 복제하지 않는다. 기존 Wan inference scheduler와 일치시키기 위해 기존 TokenLight rectified-flow latent objective를 유지하고 LGI의 **물리 조건 표현만** 가져온다. LGI `[min,max,nearest,validity]`는 작은 CNN으로 target patch grid에 맞춘 dense prefix token이 되고 source와 Lightoken prefix 옆에서 self-attention한다. spatial condition dropout 기본값은 0.1이다.

관련 파일:

- shared dense-map encoder/trainer: `model/wan_spatial.py`, `model/train_tokenlight_spatial_safe.py`
- entrypoint: `model/train_tokenlight_lgi.py`
- config: `configs/train_480/rgb_spatial_lgi.json`

### 점검과 실행

현재 한 예시만 확인:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_lgi.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_lgi.json \
  --preflight \
  --preflight_scene_id scene_000001 \
  --preflight_max_samples 1
```

LGI를 모두 채운 뒤 전체 확인:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_lgi.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_lgi.json \
  --preflight
```

single GPU 학습:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_lgi.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_lgi.json
```

ZeRO-3 학습:

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_tokenlight_lgi.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/rgb_spatial_lgi.json
```

## 4. GT direct-lit/cast-shadow mask oracle

이 경로가 encoder에 넣는 것은 다음 두 채널뿐이다.

```text
channel 0: object direct-lit geometry (`inf_mask`)
channel 1: receiver cast shadow (`shadow_mask`)
```

예를 들어 `position_002_masks` 아래의 두 파일은 다음처럼 연결된다.

```text
object_direct_lit_geometry_ray_clean_minarea00.png -> direct-lit channel
object_shadow_geometry_ray_clean_minarea00.png    -> cast-shadow channel
```

object mask는 shared `masks/object_mask.png`에서 읽지만 encoder condition에는 넣지 않고 아래 관계를 검증하는 데만 쓴다. geometry-ray 두 PNG는 원본이 정확한 binary 0/255인지도 확인한다.

- `direct_lit`은 `object`의 부분집합
- `cast_shadow`와 `object`는 서로 겹치지 않음
- `direct_lit`과 `cast_shadow`는 서로 겹치지 않음

직접광이나 cast shadow가 전혀 없는 all-zero map도 물리적으로 가능한 유효 GT다. learned presence embedding이 “유효하지만 두 채널이 0인 map”과 “조건이 없거나 dropout된 상태”를 구분한다.

### 이 실험의 해석상 주의점

`direct_lit`과 `cast_shadow`는 target light와 정답 geometry ray tracing으로 계산되어 결과의 위치를 거의 직접 알려준다. 따라서 이 결과는 일반 inference 성능이 아니며 LGI/PBR/기본 모델과 공정한 방법 비교로 쓰면 안 된다. 목적은 다음 두 가지로 제한한다.

- spatial condition encoder와 Wan이 완전한 물리 힌트를 활용할 수 있는지 확인
- 예측 condition의 품질과 무관한 모델 capacity 상한선을 측정

실제 inference에서 같은 ray-traced map을 외부 renderer가 제공할 수 없다면 이 모델은 그대로 배포할 수 없다. 반대로 oracle도 실패한다면 condition 표현/주입 또는 optimization 문제를 먼저 의심해야 한다.

관련 파일:

- entrypoint: `model/train_tokenlight_gt_masks.py`
- shared implementation: `model/train_tokenlight_spatial_safe.py`, `model/wan_spatial.py`
- config: `configs/train_480/rgb_spatial_gt_masks.json`

scene 하나의 32개 position을 점검하면 `position_002`도 포함된다.

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_gt_masks.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_gt_masks.json \
  --preflight \
  --preflight_scene_id scene_000001 \
  --preflight_max_samples 32
```

전체 점검:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_gt_masks.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_gt_masks.json \
  --preflight
```

single GPU 학습:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_gt_masks.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_gt_masks.json
```

ZeRO-3 학습:

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_tokenlight_gt_masks.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/rgb_spatial_gt_masks.json
```

## single GPU, ZeRO-3, global batch

모든 trainer는 `--train_mode single` 경로를 제공한다. single 모드는 FP32 master trainable parameter와 BF16 autocast를 사용하고, non-DeepSpeed 경로에서 gradient clipping을 수행한다. `zero3` 모드는 실제 Accelerate DeepSpeed plugin이 없으면 즉시 실패하도록 되어 있다.

optimizer 한 번이 보는 effective global batch는 다음과 같다.

```text
effective global batch
  = per-GPU batch_size
  * gradient_accumulation_steps
  * number of processes/GPUs
```

현재 config 기본값은 다음과 같다.

| 경로 | per-GPU batch | accumulation | GPU가 N개일 때 effective batch |
|---|---:|---:|---:|
| PBR | 1 | 40 | `40N` |
| LGI | 1 | 24 | `24N` |
| GT masks | 1 | 24 | `24N` |
| MultiShadow diffusion | 1 | 8 | `8N` |
| box predictor | 8 | 1 | `8N` |

GPU 수를 늘리면서 동일한 optimizer dynamics를 비교하려면 accumulation을 반비례해서 줄여 global batch를 고정한다. 예를 들어 PBR global batch 40을 GPU 5개에서 유지하려면 per-GPU batch 1, accumulation 8을 쓴다.

`configs/accelerate_zero3.yaml`의 기본 `num_processes`는 1이므로 실제 multi-GPU 실행에서는 위 명령처럼 `--num_processes N`을 명시해야 한다.

## 네 실험을 공정하게 비교하는 순서

1. 동일 reject list를 적용한다.
2. scene 단위 train/val/test split을 공유한다. 현재 MultiShadow builder는 scene-disjoint split을 만들지만 PBR/LGI/GT 기본 config는 전체 filtered metadata를 가리키므로, 이 상태의 metric은 직접 비교하면 안 된다.
3. PBR/LGI/GT도 비교 실험에서는 `data_train/objaverse_fixed32_coshadow_480/train.jsonl`을 동일한 train manifest로 사용하고, cache spec의 metadata 쪽도 함께 바꾼다. 한쪽만 바꾸면 cache dataset이 원래 manifest를 다시 읽을 수 있다.
4. base checkpoint, LoRA rank, 학습 optimizer step 수, effective global batch, seed를 맞춘다.
5. 동일한 held-out test scene과 동일한 inference seed/step/CFG를 사용한다.
6. 전체 RGB metric과 함께 object 영역 및 cast-shadow 영역 metric을 별도로 낸다. oracle GT-mask 결과는 upper bound로 따로 표기한다.

추천 ablation 순서는 `latent-only baseline -> PBR -> LGI -> predicted-box MultiShadow -> GT-box MultiShadow -> GT-mask oracle`이다. 이 순서면 intrinsic prior, 계산된 physics prior, 예측 layout, 정답 layout, 정답 dense mask의 기여를 분리할 수 있다.

## 현재 준비 상태와 남은 제한

- 이 문서와 코드는 장시간 학습 결과를 주장하지 않는다. dataset/preflight와 구조 검증 뒤 실제 convergence 및 inference 품질을 별도로 확인해야 한다.
- LGI는 현재 `scene_000001/position_00` 예시만 있으므로 나머지 map 생성이 완료되기 전에는 전체 학습할 수 없다. LGI 생성기 자체는 이 네 trainer에 포함되지 않고, 위 알고리즘과 파일 계약에 맞춘 external preprocessing이 필요하다.
- MultiShadow는 먼저 metadata builder를 실행하고 box predictor를 학습해야 한다. `gt` bbox diffusion은 diagnostic이고 최종 비교는 `predicted` bbox를 사용해야 한다.
- fixed32는 원 논문의 target-light shadow-free composite가 아니므로 MultiShadow 결과를 논문 재현 성능이라고 부르면 안 된다.
- Wan adaptation에는 exact GAAM, CLIP text positional-token injection, cross-attention KL alignment가 없다. 특히 `attention_alignment_weight`는 의도적으로 unsupported다.
- PBR은 현재 depth/normal을 매 batch VAE encode한다. safe trainer의 per-stream clean `t=0` modulation과 맞추기 위해 inference는 반드시 신규 `scripts/infer_manifest_pbr_safe.py`를 사용한다. 이 wrapper는 신규 safe model function만 교체하고 기존 inference I/O 구현과 기존 파일은 그대로 보존한다.
- LGI/GT-mask/MultiShadow용 완성된 inference entrypoint는 이 학습 구현 범위에 포함되지 않는다. checkpoint를 평가하려면 학습 때와 같은 prefix encoder/layout module과 condition packing을 inference에도 연결해야 한다.
- dense map과 PBR condition은 학습뿐 아니라 해당 모드의 inference에서도 필요하다. 특히 GT-mask oracle은 정답 map이 없으면 사용할 수 없다.

이 제한을 유지한 채 실험하면 네 경로의 결론은 각각 “joint intrinsic이 도움이 되는가”, “예측 shadow layout이 도움이 되는가”, “light-aware 2.5D occlusion prior가 도움이 되는가”, “완전한 정답 spatial prior를 주었을 때 가능한 상한은 어디인가”로 해석할 수 있다.
