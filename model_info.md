# TokenLight 물리 조건부 relighting 구현·데이터·실험 종합 문서

## 0. 2026-08-12 논문 재점검 후 v2 정정 (이 절이 아래 구버전 설명보다 우선)

세 논문 원문과 현재 코드를 다시 대조해 다음 불일치를 수정했다. 아래 본문의
`latent-only`, LGI 4채널, CoShadow self-attention layout prefix 및 attention alignment
미지원 설명은 v1 기록이며 더 이상 현재 코드 계약이 아니다.

- **PBR / UniRelight**: 논문의 핵심인 temporal/token concatenation, modality type,
  independent noise, albedo loss 0.1, 그리고 70/18/12 conditioning schedule은 기존 safe
  구현에 이미 반영돼 있었다. Wan과 numeric Lightoken을 유지하는 조건에 따라 albedo를
  depth/normal 두 modality로 교체한다. 현재 loss weight는 각각 0.1로 설정한다.
- **CoShadow / MultiShadow**: 4개 bbox token을 image self-attention prefix에 넣던 v1을
  폐기했다. v2는 source/object-mask/light를 image prefix로 유지하면서, 학습 가능한
  `object`, `casting shadow` semantic token과 `[x1,y1,x2,y2]` token을 Wan text
  cross-attention context에 추가한다. 네 positional token의 공간 attention과 GT shadow
  distribution 사이 KL alignment loss도 활성화한다. config는 논문과 맞춰 diffusion
  LR `1e-5`, batch/accumulation 4, 80 epochs로 변경했다.
- **LGI**: local validity를 제거하고 논문 Eq. 8의 `[min,max,nearest-to-zero]` 정확히
  3채널만 사용한다. 일반 noise-to-data FlowMatch 대신 source latent에서 target latent로
  가는 Brownian bridge와 drift MSE를 사용한다. `z1_hat=z(t)+(1-t)v_theta`를 frozen Wan
  VAE로 decode하고, source/target brightness difference `>0.01` 영역을 17×17 dilation한
  weighted L1을 `lambda=10`으로 더한다. 기본 bridge noise sigma는 `0.1`이며 실험 시
  반드시 기록해야 한다.
- **GT direct/shadow masks**: 논문 재현 모델이 아니라 의도된 oracle upper bound이므로
  `[direct_lit, cast_shadow]` dense condition + 기존 Wan FlowMatch를 그대로 유지한다.

Wan 2.2 TI2V 5B와 기존 numeric `LightokenEncoder`는 네 경로에서 보존했다. 단,
CoShadow 원 논문의 SD1.5 UNet multi-scale GAAM과 frozen CLIP 원어휘를 Wan에 그대로
복사할 수는 없으므로, v2의 cross-attention token 및 alignment head는 해당 원리를
Wan block 인터페이스에 옮긴 재현이다. fixed32는 단일 object/point-light 데이터이므로
논문의 multi-object association 성능을 주장할 수 없다.

> 최종 점검 기준: 2026-08-13 (Asia/Seoul)  
> 저장소: `/workspace` (`/media/ssd1/users/jaeho/relighting`)  
> 검증 컨테이너: `f8c4e2e879bd` (`jaeho_relighting`)  
> 이 문서는 지금까지 구현하고 확인한 모델, 데이터, loss 설계, 실행 방법,
> 검증 결과와 남은 제한을 한곳에 정리한 구현 명세서다.

## 1. 가장 먼저 알아야 할 결론

현재 연구용으로 준비한 핵심 경로는 다음 네 가지다.

1. **PBR / UniRelight adaptation**
   - UniRelight의 albedo 공동 예측을 depth와 normal 공동 예측으로 바꾼다.
   - RGB, depth, normal을 한 Wan DiT sequence에서 condition 또는 target stream으로 처리한다.
2. **CoShadow / MultiShadow adaptation**
   - 먼저 target light에 따른 cast-shadow bounding box와 shadow presence를 예측한다.
   - 예측한 box를 4개의 layout token으로 바꿔 TokenLight/Wan RGB 생성에 넣는다.
3. **LGI adaptation**
   - 논문의 Light-Geometry Interaction map `[min, max, nearest-to-zero]`를
     camera-aligned dense condition으로 넣는다.
4. **GT direct-lit/cast-shadow oracle**
   - `direct_lit`과 `cast_shadow` GT mask 두 채널을 그대로 condition으로 넣어
     완전한 공간 힌트가 있을 때 가능한 상한을 측정한다.

CoShadow는 box predictor와 diffusion을 따로 학습하므로, 실제 학습 프로세스 수는
다섯 개다.

| 연구 질문 | 학습 entrypoint | 핵심 조건/출력 |
|---|---|---|
| PBR 공동 학습이 RGB relighting에 도움이 되는가? | `model/train_tokenlight_pbr_safe.py` | RGB + depth + normal 공동 stream |
| target light에 따른 shadow 위치를 예측할 수 있는가? | `model/train_coshadow_box_predictor.py` | bbox + presence |
| 예측 shadow layout이 RGB 생성에 도움이 되는가? | `model/train_tokenlight_coshadow.py` | source/object/layout/light → RGB + mask logits |
| 계산된 light-geometry prior가 도움이 되는가? | `model/train_tokenlight_lgi.py` | LGI dense map condition |
| 완전한 GT 공간 prior를 주면 어디까지 가능한가? | `model/train_tokenlight_gt_masks.py` | direct-lit + cast-shadow condition |

중요한 공통 결정은 다음과 같다.

- 네 Wan 기반 학습은 기본적으로 **latent rectified-flow velocity loss만** 사용한다.
- RGB decoder image loss는 모두 `0`이다.
- 매 학습 iteration에서 전체 inference trajectory를 끝까지 unroll하지 않는다.
- 기존 `model/train.py`와 기존 PBR/추론 파일은 수정하지 않았다.
- 새 기능은 별도 trainer/model/config/wrapper 파일로 추가했다.
- single GPU에서는 FP32 trainable parameter와 BF16 autocast를 사용한다.
- ZeRO-3에서는 DeepSpeed가 low-precision shard와 FP32 master state를 관리한다.

이 네 구현은 각 논문의 **exact reproduction이 아니다**. 기존 Wan 2.2 TI2V 5B,
TokenLight numeric light token, fixed32 데이터 계약과 rectified-flow scheduler를
유지한 adaptation이다.

공식 참고 자료:

- [UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting](https://papers.nips.cc/paper_files/paper/2025/hash/9bb159d886a92a8c3375dc6cbb383333-Abstract-Conference.html)
- [UniRelight NVIDIA project](https://research.nvidia.com/labs/toronto-ai/UniRelight/)
- [MultiShadow: Multi-Object Shadow Generation for Image Compositing via Diffusion Model](https://arxiv.org/abs/2603.02743)
- [Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps](https://arxiv.org/abs/2602.21820)

## 2. 구현 원칙과 기존 코드 보존

사용자 요청에 따라 기존 구현을 덮어쓰지 않고 새 파일을 추가하는 방식을 사용했다.
새 trainer가 기존 코드에서 재사용하는 것은 dataset, Wan pipeline, LoRA helper,
TokenLight helper 같은 읽기 가능한 공통 기능뿐이다.

### 2.1 핵심 신규 파일 목록

공통 runtime:

- `model/tokenlight_physics_runtime.py`

PBR:

- `model/tokenlight_wan_pbr_safe.py`
- `model/train_tokenlight_pbr_safe.py`
- `scripts/infer_manifest_pbr_safe.py`
- `configs/train_480/fixed32_pbr_depth_normal.json`

CoShadow/MultiShadow:

- `scripts/build_fixed32_coshadow_metadata.py`
- `model/coshadow_box_predictor.py`
- `model/train_coshadow_box_predictor.py`
- `model/tokenlight_wan_coshadow.py`
- `model/train_tokenlight_coshadow.py`
- `configs/train_480/coshadow_box.json`
- `configs/train_480/coshadow_diffusion.json`
- `tests/test_coshadow.py`
- `docs/COSHADOW_FIXED32.md`

LGI 및 GT-mask 공통 spatial 경로:

- `model/wan_spatial.py`
- `model/train_tokenlight_spatial_safe.py`
- `model/train_tokenlight_lgi.py`
- `model/train_tokenlight_gt_masks.py`
- `configs/train_480/rgb_spatial_lgi.json`
- `configs/train_480/rgb_spatial_gt_masks.json`

종합 문서:

- `docs/PHYSICS_CONDITIONED_TRAINING.md`
- `model_info.md` (현재 문서)

decoder 안정화 관련 별도 신규 경로도 존재한다.

- `model/train_decoder_safe.py`
- `model/decoder_space_loss.py`

다만 네 물리 조건부 기본 config는 이 decoder loss를 사용하지 않는다.

### 2.2 기존 파일을 그대로 둔 이유

기존 RGB latent-only 학습은 과거 성공 경로이므로, 새 objective나 timestep 의미를
그 파일 안에서 조건 분기로 섞으면 다음 문제가 생긴다.

- decoder weight가 0이어도 RNG 호출 순서나 dtype 변환이 달라질 수 있다.
- PBR stream 수에 따라 type embedding의 행 의미가 바뀔 수 있다.
- one-frame prefix에 잘못된 timestep modulation이 broadcast될 수 있다.
- 기존 config가 모르는 새 key를 조용히 버리는 경우가 있어 실제 실험과 기록이
  달라질 수 있다.
- single GPU와 ZeRO-3이 서로 다른 optimizer precision을 필요로 한다.

따라서 baseline은 보존하고 새 연구 기능은 별도 entrypoint로 격리했다.

## 3. 공통 데이터 상태

### 3.1 기준 경로

```text
원본 PNG root
  data/objaverse_fixed32_png

기본 metadata
  data_train/objaverse_fixed32_480/metadata.jsonl

reject 목록
  data/objaverse_fixed32_png/reject_metadata.txt

RGB VAE latent cache
  data/vae_cache_fixed32_480/rgb

shadow-mask VAE latent cache
  data/vae_cache_fixed32_480/shadow_mask

LGI root
  data/objaverse_32_lgimap

CoShadow 파생 metadata
  data_train/objaverse_fixed32_coshadow_480
```

### 3.2 fixed32 metadata 현황

2026-08-13 컨테이너 안에서 다시 집계한 결과다.

| 항목 | 값 |
|---|---:|
| 원본 metadata rows | 48,835 |
| 원본 scenes | 1,681 |
| reject 목록에 기록된 scenes | 523 |
| 실제 metadata와 겹쳐 제외된 scenes | 456 |
| reject 적용 후 rows | 35,614 |
| reject 적용 후 scenes | 1,225 |

아래 핵심 key는 원본 48,835행 모두에 존재한다.

| key | 의미 | 존재 행 수 |
|---|---|---:|
| `video` | target point-light RGB | 48,835 |
| `input_image` | scene source RGB | 48,835 |
| `mask` | object silhouette | 48,835 |
| `inf_mask` | object 내부 direct-lit geometry-ray mask | 48,835 |
| `shadow_mask` | receiver 위 cast-shadow mask | 48,835 |
| `pbr_depth_image` | scene depth PNG | 48,835 |
| `pbr_normal_image` | scene normal PNG | 48,835 |
| `attrs_json` | target light attributes | 48,835 |

### 3.3 한 metadata row의 의미

fixed32는 한 scene에 하나의 source와 여러 target point light sample을 가진다.

```text
scene
├── source.png                  ambient/HDRI 계열 source
├── pbr/depth.png               scene 단위 depth
├── pbr/normal.png              scene 단위 normal
└── position_NNN
    ├── position_NNN.png        target point-light RGB
    ├── attrs_json              해당 target light 정보
    └── position_NNN_masks
        ├── object mask
        ├── object_direct_lit_geometry_ray_clean_minarea00.png
        └── object_shadow_geometry_ray_clean_minarea00.png
```

`attrs_json`의 첫 light에는 최소한 다음 8개 numeric 값이 필요하다.

```text
x, y, z, r, g, b, lambda, d
```

CoShadow builder는 모든 값을 finite numeric으로 검사하고 원래 JSON을 그대로
파생 metadata에 보존한다.

### 3.4 RGB latent cache

현재 fixed32 RGB cache가 존재하며 sample preflight에서 source/target cache tensor는
다음 shape로 확인됐다.

```text
[48, 1, 30, 30]
```

즉 480×480 한 frame의 cache 표현이다. cache root 아래에는 현재 113개 파일이
존재한다. 파일 개수는 sample 수가 아니라 shard/bundle 개수이므로 113을 데이터
행 수로 해석하면 안 된다.

### 3.5 PBR 데이터 현황

reject 적용 후 35,614행에서:

- unique depth 경로: 1,225
- unique normal 경로: 1,225
- 누락 depth 행: 0
- 누락 normal 행: 0

scene마다 depth/normal 한 쌍을 여러 light position이 공유한다. 현재 PBR safe
trainer는 RGB source/target에는 cache를 쓰지만 depth/normal PNG는 iteration마다
VAE encode한다. 기능적으로는 맞지만 처리량을 높이려면 PBR 전용 latent cache가
추가로 필요하다.

### 3.6 direct-lit와 cast-shadow mask 의미

두 geometry teacher의 의미는 서로 다르다.

- `inf_mask` / `object_direct_lit_geometry_ray_clean_minarea00.png`
  - object 내부에서 front-facing이고 target light까지 ray가 가려지지 않은 영역
  - 즉 직접 조명을 받을 수 있는 object geometry 영역
- `shadow_mask` / `object_shadow_geometry_ray_clean_minarea00.png`
  - object가 light ray를 가려 receiver/floor/wall에 생긴 cast shadow
  - object 내부는 제외

확인한 semantic relation은 다음과 같다.

```text
direct_lit ⊆ object
cast_shadow ∩ object = ∅
direct_lit ∩ cast_shadow = ∅
```

빈 direct/shadow mask는 항상 corruption이 아니다. 특정 light/camera 배치에서는
물리적으로 빈 GT가 정상일 수 있다. 예를 들어 position_002 완비 샘플 audit에서는
direct all-zero 35개, shadow all-zero 12개가 확인됐다.

PNG는 camera-aligned 480×480 조건으로 사용한다. 같은 mask의 raw NPY를 검사했을
때 PNG와 수직 방향이 반대인 사례가 일관되게 확인됐으므로, 현재 학습은 metadata의
PNG를 사용한다. NPY를 직접 사용할 경우 `flipud` 여부와 camera convention을 먼저
검증해야 한다.

### 3.7 LGI 데이터의 현재 coverage

초기에는 예시 한 scene만 있었지만 2026-08-13 현재 데이터가 일부 추가됐다.

전체 LGI root 기준:

| 항목 | 값 |
|---|---:|
| LGI scene directories | 966 |
| LGI position directories | 28,032 |
| `min/max/nearest` 완전한 triplet directories | 28,032 |
| LGI root 전체 파일 수 | 약 85,062 |

그러나 reject 적용 fixed32 학습 metadata와 교집합은 아직 완전하지 않다.

| 항목 | 값 |
|---|---:|
| filtered fixed32 rows | 35,614 |
| LGI triplet이 모두 있는 rows | 20,485 |
| LGI가 있는 filtered scenes | 705 |
| LGI가 아직 없는 rows | 15,129 |
| 일부 파일만 있는 partial rows | 0 |

현재 config는 `tokenlight_spatial_allow_missing=false`이므로 전체 metadata로 학습하면
첫 누락 LGI에서 의도적으로 중단된다. 해결 방법은 둘 중 하나다.

1. 남은 15,129행의 LGI를 모두 생성한다. 이것이 계획된 최종 경로다.
2. 임시 실험에서는 LGI가 완전한 20,485행만 별도 scene-safe manifest로 만든다.

missing을 zero map으로 조용히 대체하면 모델이 “물리적으로 zero인 LGI”와
“파일이 없는 sample”을 혼동하므로 기본값은 strict error다.

## 4. 공통 모델과 rectified-flow 수식

### 4.1 기본 모델

네 diffusion 경로의 기본 구성은 다음과 같다.

- backbone: Wan 2.2 TI2V 5B
- adaptation: LoRA rank 128
- LoRA target modules: `q,k,v,o,ffn.0,ffn.2`
- light condition: numeric `attrs_json`을 처리하는 `LightokenEncoder`
- source condition: source RGB VAE latent tokens
- target: target RGB VAE latent tokens
- 해상도: 480×480
- frames: 1
- CFG condition dropout: 기본 0.1
- gradient checkpointing: 활성화

### 4.2 rectified-flow 상태

clean latent를 `x0`, Gaussian noise를 `epsilon`, scheduler sigma를 `sigma`라고 하면
현재 학습 상태는 다음과 같다.

```text
x_t = (1 - sigma) * x0 + sigma * epsilon
v_target = epsilon - x0
```

모델은 sampled timestep에서 `v_pred`를 예측하고 기본 RGB loss는 다음이다.

```text
L_rgb = w_scheduler(t) * MSE(v_pred, v_target)
```

PBR target stream도 같은 sigma를 사용하지만 stream마다 독립 noise를 샘플링한다.
LGI와 GT mask는 noisy prediction target이 아니라 clean condition prefix다.

### 4.3 condition prefix와 target timestep

중요한 구현 규칙은 condition token과 target token의 timestep 의미를 분리하는 것이다.

```text
source / mask / spatial / layout / light condition → clean timestep t=0
RGB 또는 PBR noisy target                        → sampled timestep t
```

Wan의 one-frame 경로는 원래 timestep modulation이 `[B,6,D]`라 모든 token에 같은
값이 broadcast되기 쉽다. 새 PBR/spatial/CoShadow model function은 이를
`[B,L,6,D]`로 확장해 token segment마다 clean/noisy modulation을 명시한다.

이 구분이 없으면 clean depth/normal, LGI, mask 또는 light condition이 noisy target과
동일 timestep으로 해석되는 semantic mismatch가 생긴다.

## 5. decoder loss 문제와 현재 선택

이 절은 이전 RGB decoder 실험에서 왜 결과가 무너질 수 있었는지와, 네 신규 실험에서
왜 decoder를 끈 상태로 시작하는지를 설명한다.

### 5.1 문제가 된 과거 proxy

과거 decoder 경로에는 다음 형태가 있었다.

```text
pred_x0 = noise - v_pred
```

perfect prediction이면 `noise - v_target = x0`이므로 값만 보면 맞아 보이지만,
prediction error에 대한 gradient scaling은 inference와 일치하지 않는다.

올바른 sampled-timestep one-step clean estimate는:

```text
x0_pred = x_t - sigma * v_pred
```

이다. 두 error는 다음처럼 다르다.

```text
(noise - v_pred) - x0
  = v_target - v_pred

(x_t - sigma*v_pred) - x0
  = sigma * (v_target - v_pred)
```

즉 과거 식은 low-sigma 영역에서 실제 trajectory가 받아야 할 영향보다 velocity error를
과도하게 decoder까지 전달할 수 있다. 밝기 saturation, 평균색 쏠림, 디테일 뭉개짐의
유력한 원인이 된다.

또 latent loss와 decoder loss를 먼저 더한 뒤 velocity용 scheduler weight를 전체에
곱하면, image-space loss까지 velocity weighting을 강제로 공유하게 된다. 두 objective는
분리하는 것이 안전하다.

### 5.2 전체 denoising을 unroll해야 하는가

최종 inference image에 대한 정확한 image loss를 원하면 sampled timestep에서 최종
`t=0`까지 모든 denoising step을 펼치고 VAE decode한 뒤 전체 trajectory를
backpropagate해야 한다. 5B DiT에서 40~50 step을 그대로 unroll하면 계산량과 VRAM이
매우 커진다.

그래서 현실적인 선택지는 다음 세 가지다.

1. latent velocity loss만 사용한다.
2. `x_t - sigma*v_pred` 같은 one-step clean proxy를 사용한다.
3. sampled timestep의 current-flow state를 decode해 local metric loss를 건다.

네 물리 조건부 학습은 가장 안정적인 1번으로 시작한다.

### 5.3 별도 safe decoder trainer의 현재 objective

`model/train_decoder_safe.py`는 기존 baseline을 건드리지 않는 opt-in
실험이다. 현재 구현 version은 `tokenlight_decoder_current_t_v3`이고 다음을 비교한다.

```text
D(x0 + sigma*v_pred)
vs
D(x0 + sigma*v_target) = D(x_t)
```

이는 최종 clean-image reconstruction loss가 아니라 sampled timestep의 flow-state를
VAE decoder metric으로 비교하는 local consistency loss다.

안전 장치:

- decoder weight가 0이면 원본 `FlowMatchSFTLoss`에 delegate
- scheduler index를 authoritative FP32 index로 보존
- latent scheduler weight와 decoder outer weight 분리
- warmup/ramp 지원
- frozen VAE parameter에는 gradient 없음
- prediction latent에는 gradient 유지
- DiffSynth wrapper의 최종 `clamp_(-1,1)`을 피하고 raw VAE model decode
- VAE decoder gradient checkpointing 가능
- stateful causal VAE cache를 decode 후 clear
- target decode는 `no_grad`, prediction decode만 autograd
- single GPU trainable parameter/Adam state FP32

이 경로는 연구용이며 현재 네 PBR/CoShadow/LGI/GT 기본 config와 섞지 않는다.

## 6. PBR / UniRelight depth-normal adaptation

### 6.1 연구 목적

UniRelight의 핵심 아이디어는 relit RGB와 intrinsic decomposition을 하나의 DiT
sequence에서 공동으로 학습해, relighting 생성과 scene 물성 이해가 서로 도움을
주게 만드는 것이다. 원 논문은 주로 albedo를 공동 stream으로 사용한다.

현재 adaptation은 albedo 하나를 다음 두 stream으로 바꾼다.

```text
depth
normal
```

질문은 다음과 같다.

> RGB만 예측하는 것보다 depth와 normal을 condition/target으로 함께 다루면
> geometry-aware relighting이 좋아지는가?

### 6.2 논문과 같은 부분, 다른 부분

같은 핵심:

- source와 relit RGB, intrinsic stream을 하나의 transformer sequence로 처리
- modality/role별 type embedding 사용
- intrinsic을 예측하는 mode와 GT intrinsic을 condition으로 주는 mode 혼합
- source를 제거하고 intrinsic만으로 RGB를 예측하는 mode 포함
- intrinsic auxiliary loss 총 budget 0.1

변경한 부분:

- albedo 대신 depth와 normal 사용
- albedo 대신 depth와 normal을 사용하며 loss weight는 각각 0.1
- environment map encoder 대신 기존 numeric Lightoken 사용
- UniRelight 원 backbone 대신 Wan 2.2 TI2V 5B 사용
- video 대신 fixed32 1-frame image 사용
- 원 논문의 전체 objective 대신 rectified-flow latent velocity MSE만 사용
- decoder/image-space loss 사용 안 함

따라서 결과를 “UniRelight 재현”이 아니라 “UniRelight식 joint stream을 적용한
depth-normal TokenLight ablation”으로 표기해야 한다.

### 6.3 세 가지 sample mode

각 sample은 독립적으로 다음 mode 중 하나가 된다.

| 확률 | source | depth/normal 역할 | RGB 역할 |
|---:|---|---|---|
| 0.70 | condition | noisy target | noisy target |
| 0.18 | condition | clean condition | noisy target |
| 0.12 | zero/drop | clean condition | noisy target |

코드상 mode sampling은 `TokenLightPbrSafeTrainingModule._sample_unirelight_modes`에서
batch sample마다 `torch.rand`로 수행한다.

#### 70% joint-target mode

```text
condition: source + light
target:    RGB + depth + normal
```

RGB, depth, normal 모두 velocity를 예측한다. depth와 normal은 RGB와 같은 sigma를
사용하지만 각각 독립 Gaussian noise를 받는다.

#### 18% GT-PBR-condition mode

```text
condition: source + clean depth + clean normal + light
target:    RGB
```

depth/normal token은 clean latent와 condition type embedding, `t=0` modulation을
사용한다. 이 sample의 depth/normal velocity loss는 0이다.

#### 12% source-drop mode

```text
condition: clean depth + clean normal + light
source:    zero latent
target:    RGB
```

source latent를 zero로 바꾸지만 RGB latent supervision은 weight 1.0으로 유지한다.
decoder나 illumination auxiliary head가 없기 때문에 source-drop에서도 RGB loss를
끄면 학습 신호가 사라진다.

### 6.4 PBR type embedding schema

두 stream일 때 type embedding semantic layout은 다음과 같다.

```text
0: source
1: light
2: RGB target
3: depth target
4: depth condition
5: normal target
6: normal condition
```

target/condition 역할이 다르면 같은 depth라도 다른 type row를 사용한다. stream을
추가할 때 target/condition pair가 두 행씩 늘어난다.

기존 RGB TokenLight checkpoint의 4행 schema는 다음이었다.

```text
0: source
1: mask
2: light
3: RGB target
```

PBR safe trainer는 shape가 4행인 legacy checkpoint를 다음처럼 명시적으로
migration한다.

```text
PBR source row       <- RGB source row 0
PBR light row        <- RGB light row 2
PBR RGB target row   <- RGB target row 3
depth target row     <- RGB target row 3
normal target row    <- RGB target row 3
depth condition row  <- 새 초기값
normal condition row <- 새 초기값
```

condition row는 legacy RGB schema에 대응 의미가 없어 random initialization을
유지한다. 이미 PBR schema와 정확히 같은 shape면 그대로 load한다. 그 외 알 수 없는
shape는 조용히 건너뛰지 않고 오류를 낸다.

### 6.5 token sequence와 timestep

실제 sequence 순서는 다음이다.

```text
[source]
[RGB target]
[depth target 또는 condition]
[normal target 또는 condition]
[light]
```

각 latent stream은 Wan patchify를 거쳐 동일 spatial grid token이 된다. depth와
normal도 현재는 RGB Wan VAE를 사용하므로 RGB target과 latent grid가 정확히 같아야
한다.

timestep modulation은 role에 따라 분리한다.

```text
source                      → t=0
RGB target                  → sampled t
depth/normal target         → sampled t
depth/normal condition      → t=0
light                       → t=0
```

이 역할 분리는 `model/tokenlight_wan_pbr_safe.py`에서 rank-3 one-frame modulation을
tokenwise rank-4 modulation으로 확장해 구현했다. 기존 `model/tokenlight_wan_pbr.py`는
수정하지 않았다.

### 6.6 noise와 loss 수식

RGB:

```text
x_t_rgb = (1-sigma)*x0_rgb + sigma*epsilon_rgb
v*_rgb  = epsilon_rgb - x0_rgb
```

PBR stream `s`가 target인 sample:

```text
x_t_s = (1-sigma)*x0_s + sigma*epsilon_s
v*_s  = epsilon_s - x0_s
```

PBR stream이 condition이면 model input은 `x0_s`이고 loss mask는 0이다.

sample별 velocity MSE를 먼저 구한 뒤 batch mean을 사용한다.

```text
L_unweighted
  = 1.0 * mean_i MSE(v_rgb_i, v*_rgb_i)
  + 0.1 * mean_i [I(depth_target_i) * MSE(v_depth_i, v*_depth_i)]
  + 0.1 * mean_i [I(normal_target_i) * MSE(v_normal_i, v*_normal_i)]

L = w_scheduler(t) * L_unweighted
```

condition sample을 active sample 수로 다시 나눠 loss를 키우지 않고, zero mask를
포함한 전체 batch mean을 사용한다. 이 방식은 sample이 여러 rank로 나뉘어도
objective scale을 안정적으로 유지한다.

PBR target 확률이 0.7이므로 전체 training distribution에서 depth/normal의 기대
기여는 각 `0.1 * 0.7`에 해당한다. config의 0.1을 “active target sample에서
항상 RGB의 10%만 gradient가 난다”는 의미로 단순 해석하면 안 된다.

### 6.7 데이터 로딩

config:

```text
dataset_base_path       = data/objaverse_fixed32_png
dataset_metadata_path   = data_train/objaverse_fixed32_480/metadata.jsonl
reject list             = data/objaverse_fixed32_png/reject_metadata.txt
RGB cache               = data/vae_cache_fixed32_480/rgb
depth metadata key      = pbr_depth_image
normal metadata key     = pbr_normal_image
```

RGB source/target는 cache를 읽고, depth/normal은 raw PNG를 RGB로 열어 VAE encode한다.
현재 35,614 filtered row 및 1,225 unique scene의 depth/normal 경로가 모두 존재한다.

주의할 점:

- depth PNG가 어떤 metric scale/encoding을 사용하는지 trainer가 의미적으로
  판별하지 않는다.
- normal이 world/view space인지, OpenGL/OpenCV convention인지 trainer가 판별하지
  않는다.
- 데이터 전체에서 encoding과 orientation이 일관돼야 한다.
- RGB VAE는 signed physical scalar를 위해 설계된 encoder가 아니므로 이는 편리한
  첫 adaptation이지 최적의 depth/normal representation이라고 단정할 수 없다.

### 6.8 현재 PBR config

`configs/train_480/fixed32_pbr_depth_normal.json`:

| 항목 | 값 |
|---|---|
| resolution / frames | 480×480 / 1 |
| batch per GPU | 1 |
| gradient accumulation | 40 |
| single-GPU effective batch | 40 |
| epochs | 10 |
| LR | 1e-4 |
| weight decay | 0.01 |
| max grad norm | 1.0 |
| LoRA rank | 128 |
| LoRA targets | q,k,v,o,ffn.0,ffn.2 |
| CFG dropout | 0.1 |
| RGB loss weight | 1.0 |
| depth / normal weight | 0.1 / 0.1 |
| RGB/PBR decoder weight | 0 / 0 |
| save interval | 8,000 optimizer steps |

35,614행을 single GPU에서 10 epochs 돌리면 accumulation rounding 기준 약 8,910
optimizer updates다. `save_steps=8000`이므로 중간 portable checkpoint가 매우 적을 수
있다.

### 6.9 PBR checkpoint와 inference

portable export에는 최소 다음 학습 component가 포함돼야 한다.

- DiT LoRA
- `light_encoder`
- PBR type embedding

PBR inference는 반드시 `scripts/infer_manifest_pbr_safe.py`를 사용해야 한다. 이
wrapper는 기존 `scripts/infer_manifest_pbr.py`의 manifest/I/O pipeline을 재사용하되,
runtime model function만 safe PBR 버전으로 교체한다. multi-GPU spawn worker에서도
safe function을 다시 설치한다.

그 이유는 학습 checkpoint가 clean condition `t=0` / noisy target sampled `t`
semantics로 학습됐기 때문이다. legacy inference model function을 사용하면 학습과
추론의 timestep 의미가 달라질 수 있다.

### 6.10 PBR 실행 명령

dataset preflight:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_pbr_safe.py \
  --train_mode single \
  --config configs/train_480/fixed32_pbr_depth_normal.json \
  --validate_dataset_only \
  --dataset_validation_samples 32
```

single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_pbr_safe.py \
  --train_mode single \
  --config configs/train_480/fixed32_pbr_depth_normal.json
```

ZeRO-3:

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_tokenlight_pbr_safe.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/fixed32_pbr_depth_normal.json
```

safe inference의 나머지 CLI 옵션은 기존 PBR inference와 동일하며, checkpoint와
manifest를 실제 실험 경로로 지정해야 한다.

### 6.11 PBR의 현재 제한

- depth/normal 실시간 VAE encode가 throughput을 낮춘다.
- physical scalar에 RGB VAE를 쓰는 representation mismatch가 있다.
- 별도 full optimizer-state universal resume는 이 safe entrypoint에 추가하지 않았다.
- decoder/image loss는 지원하지 않으며 nonzero로 주면 parser가 거부한다.
- source-drop, condition, joint-target의 실제 validation metric을 mode별로 따로
  집계하는 evaluator는 아직 없다.
- 일반 RGB inference와 GT-PBR-conditioned inference를 혼합해 하나의 점수로 보고하면
  안 된다.

## 7. CoShadow / MultiShadow adaptation

### 7.1 연구 목적과 이름

요청 당시 CoShadow로 부른 2026 작업은 현재 arXiv에서 MultiShadow라는 제목으로
공개돼 있다. 문서와 파일명에는 요청 명칭을 유지해 `coshadow`를 사용한다.

원 논문의 큰 아이디어는 shadow-free composite/object condition과 shadow 위치
layout을 함께 이용해 plausible cast shadow를 생성하는 것이다. 현재 fixed32 adaptation은
다음 질문에 집중한다.

> target light에 따른 cast-shadow 위치를 먼저 예측하고, 그 layout을 TokenLight/Wan에
> 명시하면 RGB relighting과 shadow 영역 예측이 좋아지는가?

### 7.2 exact reproduction이 아닌 이유

fixed32의 `source.png`는 target point light에서 shadow만 제거한 composite가 아니다.
ambient/HDRI 계열 source이고, 같은 source가 최대 32개 target light row에서 반복된다.
따라서 source와 object mask만 보고 어느 target light의 shadow bbox인지 결정할 수 없다.

그래서 다음 adaptation을 적용했다.

- box predictor에 numeric target light attrs를 추가
- CLIP text positional token 대신 Wan self-attention prefix에 4개 learned layout token
  추가
- 원 논문의 exact GAAM/cross-attention alignment 대신 latent FlowMatch + shadow-mask
  auxiliary head 사용
- target-light shadow-free composite가 없으므로 background preservation 기본 off

결과는 MultiShadow 논문 성능 재현이 아니라 MultiShadow-inspired TokenLight
ablation으로 해석해야 한다.

### 7.3 전체 2단계 pipeline

```text
기존 fixed32 shadow mask
        │
        ├─ metadata builder: GT bbox / presence / 16-bin 좌표 생성
        │
source + object mask + target light attrs
        │
        ▼
1단계 CoShadowBoxPredictor
        │
        ├─ normalized XYXY bbox
        └─ shadow presence probability
        │
        ▼
16-bin quantization → x1/y1/x2/y2 layout token 4개
        │
source latent + object-mask latent + layout + light + noisy target
        │
        ▼
2단계 TokenLight/Wan
        ├─ RGB velocity
        └─ shadow mask logits
```

### 7.4 metadata builder

`scripts/build_fixed32_coshadow_metadata.py`는 새 RGB나 새 shadow image를 render하지
않는다. 기존 PNG를 완전히 검증하고 `shadow_mask`에서 파생 label을 추가한 JSONL을
생성한다.

각 row에서 검사하는 항목:

- `video`, `input_image`, `mask`, `shadow_mask`가 모두 존재하고 PNG인지
- PNG structure `verify()` 후 실제 pixel `load()`까지 성공하는지
- 네 이미지의 width/height가 서로 같고 480×480인지
- 경로가 dataset root 밖으로 escape하지 않는지
- `attrs_json` 첫 light의 `x,y,z,r,g,b,lambda,d`가 finite numeric인지
- 같은 scene/target path가 중복되지 않는지
- input row가 `valid=false`가 아닌지

#### bbox convention

shadow foreground는 grayscale threshold 127보다 큰 pixel로 정의한다.

```text
bbox = [x1, y1, x2_exclusive, y2_exclusive]
normalized by [W,H,W,H]
```

`x2/y2`를 half-open exclusive coordinate로 두므로 마지막 pixel까지 닿는 mask는
정확히 1.0이 된다. 빈 mask는:

```text
bbox       = [0,0,0,0]
bbox_valid = false
area       = 0
```

로 저장하며 정상 negative sample이다.

#### 16-bin quantization

각 normalized coordinate `c`는 다음처럼 양자화한다.

```text
bin = floor(clamp(c,0,1) * (16-1) + 0.5)
```

PyTorch의 bankers rounding과 metadata builder의 결과가 달라지지 않도록 box model
쪽에서도 round-half-up을 똑같이 구현했다.

#### scene split

```text
SHA256(seed:scene_id) → unit interval → 90% train / 5% val / 5% test
```

row 순서와 무관한 deterministic scene split이므로 같은 scene의 서로 다른 light가
다른 split으로 새지 않는다.

### 7.5 생성된 CoShadow metadata

출력 경로:

```text
data_train/objaverse_fixed32_coshadow_480/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── metadata.jsonl
├── rejected.jsonl
└── summary.json
```

실제 생성 결과:

| 항목 | 값 |
|---|---:|
| scanned source rows | 48,835 |
| reject로 제외된 rows | 13,221 |
| output rows | 35,614 |
| failures | 0 |
| empty shadow masks | 188 |
| train rows / scenes | 32,277 / 1,110 |
| val rows / scenes | 1,624 / 56 |
| test rows / scenes | 1,713 / 59 |
| output directory size | 약 99 MB |

각 row에 추가한 필드:

```text
coshadow_split
coshadow_bbox_xyxy
coshadow_bbox_valid
coshadow_bbox_bins
coshadow_bbox_num_bins
coshadow_shadow_area_pixels
coshadow_image_width
coshadow_image_height
```

원래 `attrs_json`과 이미지 경로는 그대로 보존한다.

### 7.6 1단계: light-conditioned shadow-box predictor

관련 파일:

- model: `model/coshadow_box_predictor.py`
- trainer: `model/train_coshadow_box_predictor.py`
- config: `configs/train_480/coshadow_box.json`

#### 입력

```text
source RGB              [B,3,256,256]
object mask             [B,1,256,256]
CoordConv x/y           [B,2,256,256]
target light attrs      LightokenEncoder input
```

visual branch는 총 6채널이다. RGB는 bicubic, mask는 nearest로 256×256 resize한다.

#### visual encoder

channel progression:

```text
6 → 32 → 64 → 128 → 256
```

각 stage는 다음 구조다.

```text
Conv3x3 stride2
GroupNorm
SiLU
Conv3x3 stride1
GroupNorm
SiLU
```

마지막 feature를 global adaptive average pooling한다.

#### light encoder와 fusion

`LightokenEncoder`가 numeric light attrs를 128차원 token으로 바꾸고 light token mean을
visual feature와 concat한다. fusion MLP는 256 hidden dimension을 사용한다.

기본 모델 parameter 수는 1,521,861개다.

| component | parameters |
|---|---:|
| visual encoder | 1,174,080 |
| light encoder | 181,632 |
| fusion MLP | 164,864 |
| bbox head | 1,028 |
| presence head | 257 |

#### 출력

```text
boxes             [B,4], normalized XYXY
presence_logits   [B]
```

bbox head raw endpoint에 sigmoid를 적용하고, x pair와 y pair를 각각 min/max 정렬해
항상 `x1<=x2`, `y1<=y2`를 만족시킨다.

#### loss

```text
L_box
  = lambda_l1       * L1(pred_box, target_box)          # valid shadow만
  + lambda_iou      * [1 - mean IoU]                    # valid shadow만
  + lambda_presence * BCEWithLogits(pred_presence, valid)
```

기본 세 lambda는 모두 1.0이다. batch에 valid shadow가 하나도 없으면 bbox L1/IoU는
graph와 연결된 zero가 되고 presence BCE만 학습한다.

logging/validation metric:

- total loss
- bbox L1
- IoU loss / mean IoU
- presence BCE / accuracy
- valid-shadow fraction

train/val scene 집합이 겹치면 학습 시작 전에 오류를 낸다.

#### optimizer/runtime

| 항목 | 값 |
|---|---|
| batch | 8 |
| accumulation | 1 |
| epochs | 100 |
| LR | 1e-4 |
| weight decay | 0.01 |
| max grad norm | 1.0 |
| mixed precision | BF16 autocast |
| parameter/Adam master | FP32 |
| scheduler | 실제 constant `LambdaLR(1.0)` |
| save interval | 1,000 optimizer steps |

32,277 train rows를 batch 8/drop-last로 100 epochs 모두 학습하면 약 403,400
optimizer updates다. 이는 상당히 긴 초기 설정이다. 실제 본 실험에서는 val IoU와
presence metric으로 early stopping/best checkpoint selection을 추가하는 것이 좋다.

checkpoint directory:

```text
outputs/train/coshadow_box_fixed32/<label>/
├── model.safetensors
└── config.json
```

기본 diffusion config는 `outputs/train/coshadow_box_fixed32/final`을 찾는다.

### 7.7 2단계: predicted layout-conditioned TokenLight/Wan

관련 파일:

- model function/layout/mask head: `model/tokenlight_wan_coshadow.py`
- trainer/loss/data attachment: `model/train_tokenlight_coshadow.py`
- config: `configs/train_480/coshadow_diffusion.json`

#### bbox source mode

두 모드를 지원한다.

```text
predicted: frozen 1단계 predictor의 bbox와 presence 사용 (기본/실전)
gt:        metadata builder의 GT bbox bin과 valid flag 사용 (oracle/smoke)
```

`predicted` mode에서는 predictor checkpoint가 없으면 학습 전에 실패한다. predictor는
`eval()`, `requires_grad_(False)`, `torch.no_grad()` 상태다. presence probability가
0.5 이상일 때 valid로 보고 bbox를 16-bin으로 바꾼다.

GT mode는 layout path 자체가 학습 가능한지 검사하거나 predicted-vs-oracle gap을
재는 용도다. 최종 일반 inference 조건에서 GT bbox가 없다면 GT mode 결과를 실제
배포 성능으로 보고하면 안 된다.

#### 4개 layout token

양자화된 bbox `[x1,y1,x2,y2]`를 정확히 네 token으로 만든다.

각 token은 다음 embedding의 합이다.

```text
x coordinate 또는 y coordinate embedding
+ coordinate slot embedding (x1/y1/x2/y2)
+ layout type embedding
+ presence embedding
```

invalid/empty shadow는 coordinate 0을 정상 좌표로 해석하지 않고 별도 learned
`invalid_coordinate_embedding`을 사용한다. 마지막에 LayerNorm을 적용한다.

#### DiT token 순서

```text
[source latent tokens]
[object-mask latent tokens]
[x1][y1][x2][y2] layout tokens
[light tokens]
[noisy RGB target tokens]
```

- source/object/target는 Wan patchify와 spatial RoPE를 사용한다.
- layout/light는 image grid가 아니므로 constant/nonspatial frequency tensor를 사용한다.
- source/object/layout/light prefix는 모두 clean `t=0` modulation이다.
- RGB target token만 sampled timestep modulation을 사용한다.

object mask latent cache directory는 현재 config에서 `null`이다. 따라서 RGB
source/target cache는 사용하지만 object mask는 필요한 경우 VAE encode한다.

#### RGB velocity head와 shadow-mask head

DiT block을 지난 뒤 prefix token을 버리고 target hidden만 사용한다.

- 기존 Wan head: RGB velocity latent 출력
- `CoShadowLatentMaskHead`: target token마다 1-channel shadow logit 출력

mask head는 LayerNorm + Linear(`D→1`)이며 target token grid로 reshape한다. GT shadow
mask는 해당 grid 크기로 area interpolation해 BCE target으로 사용한다.

#### loss

기본 objective:

```text
L_total
  = 1.0 * w_scheduler(t) * MSE(v_rgb_pred, v_rgb_target)
  + 0.1 * weighted_BCE(shadow_logits, shadow_mask)
  + 0.0 * L_background_current_state
```

shadow positive pixel weight는 4.0이다.

optional background term은 sampled timestep에서:

```text
pred_current   = x0 + sigma*v_pred
target_current = x_t
L_background  = mean_|outside shadow| |pred_current-target_current|
```

을 계산한다. 기본값은 0이다. fixed32 source가 target-light shadow-free composite가
아니고 target light가 shadow 밖 RGB도 바꿀 수 있으므로, non-shadow 영역을 과도하게
강제하는 ablation은 조심해서 해석해야 한다.

원 논문의 attention alignment에 해당하는 정확한 Wan cross-attention map extraction은
구현하지 않았다. `coshadow_attention_alignment_weight`가 0이 아니면 조용히 무시하지
않고 `NotImplementedError`로 실패한다.

### 7.8 현재 diffusion config

| 항목 | 값 |
|---|---|
| train rows | 32,277 |
| resolution / frames | 480×480 / 1 |
| bbox source | predicted |
| bbox bins | 16 |
| presence threshold | 0.5 |
| batch per GPU | 1 |
| accumulation | 8 |
| single-GPU effective batch | 8 |
| epochs | 10 |
| LR / WD | 1e-4 / 0.01 |
| LoRA rank | 128 |
| latent / mask / background weight | 1.0 / 0.1 / 0.0 |
| mask positive weight | 4.0 |
| save interval | 1,000 optimizer steps |

single GPU에서 약 40,350 optimizer updates다. 다른 spatial/PBR config보다 update 수가
크므로 현재 값 그대로는 공정한 model comparison이 아니다.

### 7.9 CoShadow 실행 순서

metadata dry-run:

```bash
python scripts/build_fixed32_coshadow_metadata.py --dry-run --limit 128
```

전체 metadata는 이미 생성돼 있다. 다시 만들 때:

```bash
python scripts/build_fixed32_coshadow_metadata.py
```

box dataset preflight:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_coshadow_box_predictor.py \
  --train_mode single \
  --config configs/train_480/coshadow_box.json \
  --validate_dataset_only \
  --validate_dataset_samples 128
```

1단계 학습:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_coshadow_box_predictor.py \
  --train_mode single \
  --config configs/train_480/coshadow_box.json
```

diffusion dataset/cache preflight:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_coshadow.py \
  --train_mode single \
  --config configs/train_480/coshadow_diffusion.json \
  --validate_dataset_only \
  --validate_dataset_samples 128
```

GT-box smoke/upper-bound:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_coshadow.py \
  --train_mode single \
  --config configs/train_480/coshadow_diffusion.json \
  --coshadow_bbox_source gt
```

predicted-box 본 학습:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_coshadow.py \
  --train_mode single \
  --config configs/train_480/coshadow_diffusion.json
```

ZeRO-3:

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_tokenlight_coshadow.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/coshadow_diffusion.json
```

### 7.10 추천 CoShadow ablation

| 실험 | bbox | mask head | 목적 |
|---|---|---|---|
| RGB latent-only baseline | 없음 | 없음 | 기본 성능 |
| layout-only GT | GT | 0 | 완벽한 coarse layout 영향 |
| layout+mask GT | GT | 0.1 | coarse layout + dense auxiliary 영향 |
| layout-only predicted | predicted | 0 | box predictor error 포함 |
| layout+mask predicted | predicted | 0.1 | 실제 메인 경로 |
| predicted + background | predicted | 0.1 + bg | background term 영향 |

box predictor 자체는 val mean IoU, presence precision/recall/accuracy와 empty-shadow
분리 metric을 보고 선택해야 한다. diffusion RGB metric만 보면 box predictor가 잘못된
원인을 분리하기 어렵다.

### 7.11 CoShadow의 현재 제한

- 새 shadow-free target-light composite 이미지는 생성하지 않았다.
- 생성한 것은 기존 shadow PNG를 참조하는 bbox/presence metadata다.
- box predictor 장시간 학습 결과와 predicted bbox 시각화는 아직 없다.
- CoShadow diffusion 장시간 학습/생성 결과도 아직 없다.
- CoShadow 전용 end-to-end inference entrypoint는 아직 없다.
- exact GAAM, CLIP positional injection, attention-alignment loss는 없다.
- box rectangle은 coarse layout이라 shadow의 정확한 곡선 형태는 mask auxiliary와
  DiT가 배워야 한다.
- 188개의 empty-shadow sample은 정상 negative로 유지되지만 별도 empty-case 평가가
  필요하다.
- box trainer의 `resume_checkpoint`는 model weight만 복원하며 optimizer/scheduler/
  epoch를 이어가는 exact resume가 아니다.
- diffusion portable checkpoint에는 frozen box predictor가 bundle되지 않는다. 추론 시
  diffusion checkpoint와 box predictor directory가 둘 다 필요하다.
- 기본 config에서는 object mask를 매 batch VAE encode하고, predicted mode는 box
  predictor를 위해 source/mask raw PNG도 매 batch 읽으므로 I/O/VAE 병목이 있다.
- CFG dropout은 Lightoken을 drop하지만 이미 light-conditioned predictor가 만든 layout
  token은 남는다. 따라서 완전히 target-light 정보가 없는 unconditional branch는 아니다.
- 현재 box validation log는 presence accuracy를 제공하지만 empty-case precision/recall/F1은
  구현하지 않았다.

## 8. LGI adaptation

### 8.1 연구 목적

LGI는 depth나 normal을 그대로 주는 대신, 현재 camera geometry와 **요청한 point
light가 상호작용할 때 어디가 가려질 가능성이 있는지**를 signed dense map으로
표현한다.

질문은 다음과 같다.

> 단순 light coordinate만 주는 것보다, light ray와 visible geometry를 결합한
> 2.5D occlusion prior를 함께 주면 cast shadow와 직접광 표현이 좋아지는가?

### 8.2 논문의 LGI map 생성 개념

pixel `(u,v)`의 depth를 `D(u,v)`, camera intrinsic을 `K`, point light를 `l`이라고
하면 먼저 pixel을 camera-space 3D point로 lift한다.

```text
p = D(u,v) * K^-1 * [u,v,1]^T
```

`p`에서 light `l`로 가는 ray 위에 기본 16개 sample을 만든다.

```text
S_n = p + delta_n * (l-p)
```

각 `S_n`을 image plane에 재투영해 observed depth surface `S'_n`과 비교한다.
camera plane normal을 `n=[0,0,1]^T`라 하면 surface ray와 light ray의 elevation
difference를 계산한다.

```text
v_s = S'_n - p
v_l = l - p

e_s = asin((v_s dot n) / ||v_s||)
e_l = asin((v_l dot n) / ||v_l||)
e_d[n] = e_s - e_l
```

sample 축에서 다음 세 값을 취한다.

```text
c1 = min_n e_d[n]
c2 = max_n e_d[n]
c3 = e_d[argmin_n |e_d[n]|]

paper channel order = [min, max, nearest-to-zero]
```

현재 trainer는 LGI를 생성하지 않는다. 외부 preprocessing이 위 계약에 맞는 NPY를
생성해야 하고 trainer는 이를 엄격하게 읽는다.

### 8.3 디렉터리 및 position mapping

필요한 형식:

```text
data/objaverse_32_lgimap/
└── scenes/
    └── scene_000001/
        ├── position_00/
        │   ├── min.npy
        │   ├── max.npy
        │   └── nearest.npy
        ├── position_01/
        └── ...
```

fixed32 metadata의 세 자리 sample name을 두 자리 LGI 디렉터리로 변환한다.

```text
position_000 → position_00
position_002 → position_02
position_031 → position_31
```

configurable 값:

- prefix: `position_`
- digits: 2
- integer offset: 0

metadata row에 `lgi_dir`이 있으면 자동 mapping보다 우선한다. 상대경로는 LGI root
기준이며 절대경로도 지원한다.

### 8.4 실제 condition channel

모델 입력은 float32 `[4,480,480]`이다.

```text
channel 0 = min
channel 1 = max
channel 2 = nearest_to_zero
channel 3 = validity
```

앞의 3개만 논문 LGI channel이다. `validity`는 로컬에서 추가한 channel이며 세 paper
channel이 모두 exact zero인 pixel을 0, 나머지를 1로 둔다.

```text
validity(u,v) = 0 if min=max=nearest=0 else 1
```

주의: 실제 물리적으로 `(0,0,0)`이 유효할 수 있는 데이터 convention이라면 이
heuristic은 틀릴 수 있다. 최종 LGI generator는 invalid sentinel과 단위를 sidecar
schema로 명시하는 편이 안전하다.

### 8.5 loader validation

`SpatialConditionReader._load_lgi`는 다음을 강제한다.

- `min.npy`, `max.npy`, `nearest.npy` 모두 존재
- `np.load(..., allow_pickle=False)`
- 각 shape가 정확히 `(480,480)`
- numeric dtype
- float32 변환 후 NaN/Inf 없음
- 모든 pixel에서 `min <= nearest <= max`
- channel stack 순서는 inequality 나열과 달리 논문 계약인
  `[min,max,nearest]`를 유지

`allow_missing=true`로 바꾸더라도 FileNotFound/KeyError만 learned null로 바뀐다.
bad shape, nonfinite, ordering violation, 잘못된 sample name은 계속 hard failure다.
즉 “결측 허용”이지 “손상 허용”이 아니다.

현재 loader가 검증하지 못하는 것:

- 값이 radian인지 다른 단위인지
- camera coordinate convention
- vertical orientation
- 해당 position의 target light와 실제로 동일한 light인지
- signed range/outlier가 논문과 동일한지

이 항목들은 LGI 생성기 provenance와 별도 alignment visualization으로 확인해야 한다.

### 8.6 SpatialConditionEncoder

LGI와 GT mask는 같은 dense-map encoder를 공유한다.

LGI 입력 설정:

```text
input channels = 4
hidden channels = 64
mean = [0,0,0,0]
std  = [1,1,1,1]
input clipping = off
```

CNN 구조:

```text
Conv7x7 stride4 → GroupNorm → SiLU
Conv3x3 stride2 → GroupNorm → SiLU
Conv3x3 stride2 → GroupNorm → SiLU
1x1 projection → Wan token dimension
adaptive average pool → target Wan spatial token grid
flatten → [B,H_token*W_token,D]
```

총 downsample stride는 16이고, adaptive pooling으로 실제 target patch grid와 정확히
맞춘다. 480×480 condition은 target spatial grid 수만큼 dense prefix token을 만든다.
이 때문에 source+target만 쓰는 baseline보다 attention sequence와 메모리 비용이
상당히 증가한다.

encoder에는 다음 learned representation도 있다.

- `null_token`
- present/missing binary embedding
- spatial condition type embedding
- future layout type embedding

정상 all-zero map은 `present=true` embedding을 받고, 파일이 없거나 dropout된 map은
learned null + `present=false` embedding을 받아 서로 구분된다.

### 8.7 token 순서와 timestep

LGI config의 실제 token 순서:

```text
[source latent]
[LGI spatial tokens]
[light tokens]
[noisy RGB target]
```

기존 generic mask token은 config에서 꺼져 있다. model function에는 future
`tokenlight_layout_tokens` 확장점이 있지만 현재 LGI reader는 layout token을 만들지
않는다.

timestep:

```text
source / LGI / light → clean t=0
RGB target           → sampled t
```

spatial condition dropout은 sample별 0.1이다. dropout 시 learned null path를 사용한다.
light CFG dropout 0.1과 spatial dropout 0.1은 서로 독립이다.

### 8.8 loss와 논문 차이

현재 loss는 오직 RGB latent FlowMatch다.

```text
L = w_scheduler(t) * MSE(v_rgb_pred, v_rgb_target)
```

- LGI reconstruction loss 없음
- shadow-mask loss 없음
- decoder image loss 없음

LGI 논문의 SDXL latent bridge matching과 decoded brightness-region L1을 복제하지
않았다. 현재 Wan inference scheduler와 성공한 TokenLight latent objective를 유지하고
LGI의 물리 condition 표현만 가져왔다.

따라서 결과는 “LGI paper full method”가 아니라 “LGI map-conditioned TokenLight”다.

### 8.9 현재 LGI config

`configs/train_480/rgb_spatial_lgi.json`:

| 항목 | 값 |
|---|---|
| batch per GPU | 1 |
| accumulation | 24 |
| effective batch, 1 GPU | 24 |
| epochs | 10 |
| LR / WD | 1e-4 / 0.01 |
| LoRA rank | 128 |
| spatial dropout | 0.1 |
| hidden channels | 64 |
| decoder loss | 0 |
| save interval | 8,000 steps |
| missing LGI | hard error |

전체 35,614행이 준비됐다고 가정하면 약 14,840 optimizer updates다. 하지만 현재
LGI가 완전한 filtered row는 20,485개뿐이므로 config를 그대로 장시간 실행할 단계는
아니다.

### 8.10 LGI 실행

특정 scene preflight:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_lgi.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_lgi.json \
  --preflight \
  --preflight_scene_id scene_000001 \
  --preflight_max_samples 1
```

전체 coverage가 채워진 뒤 full preflight:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_lgi.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_lgi.json \
  --preflight
```

single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_lgi.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_lgi.json
```

ZeRO-3:

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_tokenlight_lgi.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/rgb_spatial_lgi.json
```

### 8.11 LGI의 현재 제한

- filtered training row 15,129개 LGI가 아직 없다.
- trainer는 LGI 생성기가 아니다.
- unit/orientation/light mapping provenance가 파일에 없다.
- identity normalization이라 outlier가 그대로 encoder에 들어간다.
- LGI 전용 inference entrypoint가 아직 없다.
- 학습 때와 같은 reader/encoder/type embedding/timestep semantics를 inference에
  연결하기 전에는 checkpoint를 올바르게 평가할 수 없다.

## 9. GT direct-lit / cast-shadow oracle

### 9.1 목적

이 경로는 예측 가능한 condition을 만들기 위한 최종 모델이 아니라 진단 실험이다.

> target light에 대한 direct-lit 위치와 cast-shadow 위치를 GT로 완전히 알려줘도
> RGB relighting이 실패하는가, 아니면 잘 풀리는가?

oracle도 실패하면 PBR/LGI/box predictor보다 먼저 다음을 의심해야 한다.

- spatial encoder가 정보를 잃는지
- condition token injection이 약한지
- timestep/type embedding이 틀렸는지
- optimization/LoRA capacity가 부족한지

oracle이 잘 되고 predicted condition만 실패하면 condition predictor/data quality가
병목일 가능성이 높다.

### 9.2 정확한 모델 입력

모델 condition은 정확히 두 채널이다.

```text
channel 0 = direct_lit   ← metadata `inf_mask`
channel 1 = cast_shadow ← metadata `shadow_mask`
```

요청한 예시 파일 mapping:

```text
object_direct_lit_geometry_ray_clean_minarea00.png
  → direct_lit

object_shadow_geometry_ray_clean_minarea00.png
  → cast_shadow
```

object silhouette `mask`는 읽지만 **모델 condition에는 넣지 않는다**. 오직 semantic
relation validation에 사용한다.

### 9.3 binary 및 관계 검증

세 mask 모두 480×480이어야 한다.

- direct/shadow teacher PNG는 grayscale unique value가 정확히 0/255 subset이어야 한다.
- object mask는 antialiasing이 있어 threshold 0.5를 허용한다.
- direct/shadow는 nearest 의미의 binary tensor로 유지한다.

강제 relation:

```text
direct_lit ⊆ object
cast_shadow ∩ object = ∅
direct_lit ∩ cast_shadow = ∅
```

relation을 통과한 뒤 object channel은 버리고 `[direct,shadow]`만 stack한다.

현재 검사하지 않는 항목:

- cast shadow가 receiver mask의 부분집합인지
- 원본 RGBA 각 RGB channel/alpha가 완전히 같은지

loader는 `convert("L")` 결과를 사용한다. NPY는 지원하지 않는다.

all-zero direct 또는 shadow도 `present=true`인 정상 condition이다. missing/dropout은
learned null/presence embedding으로 구분한다.

### 9.4 encoder, token, loss

GT config의 spatial encoder는 LGI와 동일한 구조지만 입력 channel이 2다.

```text
input channels = 2
hidden channels = 64
mean/std = [0,0] / [1,1]
input clipping = off
```

token 순서:

```text
[source latent]
[direct-shadow spatial tokens]
[light tokens]
[noisy RGB target]
```

timestep:

```text
source / GT masks / light → clean t=0
RGB target                → sampled t
```

loss:

```text
L = w_scheduler(t) * MSE(v_rgb_pred, v_rgb_target)
```

mask reconstruction loss나 decoder loss는 없다. GT mask는 condition으로만 사용된다.
spatial condition dropout은 0.1이다.

### 9.5 현재 GT config

`configs/train_480/rgb_spatial_gt_masks.json`:

| 항목 | 값 |
|---|---|
| filtered rows | 35,614 |
| batch per GPU | 1 |
| accumulation | 24 |
| single-GPU effective batch | 24 |
| epochs | 10 |
| 예상 optimizer updates | 약 14,840 |
| LR / WD | 1e-4 / 0.01 |
| LoRA rank | 128 |
| spatial dropout | 0.1 |
| decoder loss | 0 |

### 9.6 GT 실행

한 scene 32 position 점검:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_gt_masks.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_gt_masks.json \
  --preflight \
  --preflight_scene_id scene_000001 \
  --preflight_max_samples 32
```

전체 preflight:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_gt_masks.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_gt_masks.json \
  --preflight
```

single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python model/train_tokenlight_gt_masks.py \
  --train_mode single \
  --config configs/train_480/rgb_spatial_gt_masks.json
```

ZeRO-3:

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  model/train_tokenlight_gt_masks.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config configs/train_480/rgb_spatial_gt_masks.json
```

### 9.7 해석상 제한

GT direct/shadow는 target geometry와 target light의 답을 거의 직접 제공한다.

- 일반 relighting 성능과 같은 표에 무표시로 넣으면 안 된다.
- 반드시 `oracle` 또는 `upper bound`로 표기한다.
- inference에도 같은 GT mask가 필요하다.
- 외부 renderer/geometry engine이 없다면 일반 image-only inference에 사용할 수 없다.
- condition dropout 10% 성능과 full oracle condition 성능을 분리해 평가해야 한다.

이 실험의 핵심 산출물은 최고 점수 자체보다 predicted condition과 oracle 사이의 gap이다.

## 10. 공통 spatial trainer 세부사항

LGI와 GT mask는 thin entrypoint만 다르고 아래 공통 구현을 사용한다.

```text
model/wan_spatial.py
model/train_tokenlight_spatial_safe.py
```

### 10.1 SpatialConditionDataset

base RGB cache dataset을 감싸되 물리 condition은 원본 metadata row에서 다시 읽는다.
base dataset이 path를 PIL로 교체하거나 선언하지 않은 key를 생략해도 LGI/mask key가
유실되지 않게 하기 위함이다.

각 item에 다음 필드를 붙인다.

```text
_tokenlight_spatial_map
_tokenlight_spatial_present
_tokenlight_spatial_info
```

condition 자체는 VAE latent cache에 넣지 않고 loader worker가 raw NPY/PNG를 읽는다.
학습 모듈은 batch stack 후 target `height,width`와 다시 shape를 비교하고 float32로
device에 옮긴다.

### 10.2 preflight

`--preflight`는 Accelerator와 5B Wan model을 만들기 전에 return하는 model-free 검사다.

- cache spec 왼쪽 metadata를 우선 사용
- `valid=false`와 reject scene 제외
- optional scene/start/max/error-limit filter
- condition tensor shape/finite 재검사
- channel mean과 resolved schema JSON 출력

단, preflight는 해당 spatial condition을 중점 검사한다. 모든 target/source PNG,
모든 VAE cache entry, light attrs 전체를 동시에 전수 검사하는 도구는 아니다.

### 10.3 checkpoint export/load

LoRA/light/type embedding 외에 다음이 portable state에 포함된다.

```text
tokenlight_spatial_encoder.*
tokenlight_spatial_type_embedding.*
```

encoder parameter뿐 아니라 channel mean/std buffer도 저장한다. load할 때 같은 key와
같은 shape인 tensor만 가져오고 skipped/missing 수를 출력한다.

LGI encoder는 4-channel 첫 conv, GT encoder는 2-channel 첫 conv이므로 두 checkpoint를
서로 완전 호환이라고 보면 안 된다. 일부 shape가 맞는 layer만 load되고 나머지는 새
초기값이 될 수 있다.

### 10.4 아직 없는 spatial inference

현재 LGI/GT 전용 inference CLI는 없다. 올바른 inference를 만들려면:

1. 동일 `SpatialConditionReader` schema와 channel order로 condition load
2. 같은 mean/std/clip/validity 적용
3. spatial encoder와 spatial type embedding checkpoint load
4. pipeline model function을 spatial 버전으로 교체
5. 매 denoising step에 spatial map/presence/drop semantics 전달
6. source/light/target과 같은 camera alignment 및 480×480 좌표 보장

이 연결 없이 일반 TokenLight inference script에 LoRA만 넣으면 새 spatial module이
로드되지 않으며 학습한 모델과 다른 network가 된다.

## 11. single GPU와 ZeRO-3 runtime

### 11.1 single GPU에서 BF16 parameter AdamW가 위험했던 이유

과거 direct `python model/train_single.py` 실행은
`configs/accelerate_single_gpu.yaml`을 읽지 않았다. Accelerator에 mixed precision을
명시하지 않으면 runtime은 `mixed_precision=no`인데, 코드가 trainable parameter를
pipeline dtype인 BF16로 바꾸고 일반 AdamW를 붙일 수 있었다.

PyTorch 일반 AdamW는 별도 FP32 master weight를 자동으로 만들지 않는다. BF16
parameter이면 momentum/variance state도 BF16이 될 수 있고, `lr=1e-4` 수준 update가
parameter 크기에 비해 양자화되어 사라질 수 있다. 이는 학습 loss가 움직여도 실제
weight update 품질을 나쁘게 만들 수 있다.

새 `model/tokenlight_physics_runtime.py` 정책:

```text
single GPU:
  trainable parameter = FP32
  optimizer state      = FP32
  forward              = BF16 autocast

ZeRO-3:
  parameter shard/master precision은 DeepSpeed config가 관리
```

### 11.2 공통 runtime 동작

- positive batch size 검사
- decoder loss nonzero 거부
- trainable parameter dtype 변환 및 count 출력
- AdamW 또는 configured optimizer 생성
- 첫 step부터 실제 constant인 `LambdaLR(lambda _: 1.0)`
- cache-aware DataLoader/base collate 재사용
- DeepSpeed microbatch/accum/global batch 설정
- `accelerator.accumulate`와 `accelerator.autocast`
- `accelerator.backward`
- non-DeepSpeed sync step에서만 grad clip
- optimizer step에서만 logger/TensorBoard step 증가
- portable model logger hook으로 checkpoint export

기존 safe launcher의 `ConstantLR` 기본 factor 1/3 시작 문제를 새 physics runtime에서는
사용하지 않는다.

### 11.3 effective global batch

```text
effective global batch
  = per-GPU micro batch
  * gradient accumulation
  * world size
```

예:

```text
PBR single: 1 * 40 * 1 = 40
LGI single: 1 * 24 * 1 = 24
GT single:  1 * 24 * 1 = 24
CoShadow:   1 *  8 * 1 = 8
```

GPU 수를 늘리고 동일 experiment dynamics를 유지하려면 accumulation을 world size에
반비례해 줄여야 한다. 예를 들어 global batch 40을 5 GPU에서 유지하려면
`micro=1, accumulation=8`이다.

### 11.4 launch mode 주의

`--train_mode zero3`라는 문자열만 plain Python에 넣는 것은 ZeRO-3 launch가 아니다.
반드시 다음처럼 실제 Accelerate DeepSpeed config를 사용해야 한다.

```bash
accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  --num_processes N \
  ENTRYPOINT.py \
  --train_mode zero3 \
  --initialize_model_on_cpu \
  --config CONFIG.json
```

CoShadow trainer는 zero3 mode인데 plugin이 없거나 single mode인데 plugin이 있으면
fail-fast한다. PBR/spatial도 항상 위 launch form을 사용해야 한다.

### 11.5 메모리 설정

5B model에서는 parameter보다 attention activation도 크다.

- PBR은 source + RGB + depth + normal spatial grid가 공동 attention한다.
- LGI/GT는 source + spatial condition + target grid가 된다.
- CoShadow는 source + object mask + target grid에 layout/light가 붙는다.

따라서 첫 안정 설정은 `microbatch=1 + accumulation`이다. batch를 5나 10으로
늘렸는데 VRAM 차이가 작아 보이는 경우는 allocator reserve, activation checkpoint,
cache/offload, 측정 시점 때문에 그럴 수 있다. 하지만 forward graph와 처리량 비용이
없어진 것은 아니다.

decoder loss를 켠 경로는 DiT graph가 살아 있는 동안 frozen VAE decoder graph도
유지하므로 microbatch 1, decoder sample subset 1, VAE decoder checkpoint가 특히
중요하다. 네 물리 config는 decoder가 off라 이 추가 graph는 없다.

## 12. 현재 config 비교와 공정한 실험 설계

### 12.1 현재 config를 그대로 실행할 때

| 모델 | 사용 rows | batch×accum | epochs | 약 optimizer updates |
|---|---:|---:|---:|---:|
| PBR | 35,614 | 1×40 | 10 | 8,910 |
| LGI* | 최대 35,614 | 1×24 | 10 | 14,840 |
| GT masks | 35,614 | 1×24 | 10 | 14,840 |
| CoShadow diffusion | 32,277 | 1×8 | 10 | 40,350 |
| CoShadow box | 32,277 | 8×1 | 100 | 403,400 |

`LGI*`: 현재 완비 row는 20,485뿐이므로 전체 config는 아직 strict missing에서
중단된다.

현재 값을 그대로 비교하면 CoShadow diffusion은 PBR보다 훨씬 많은 optimizer
updates를 받고, train split도 다르다. 이 점수를 같은 표에서 직접 비교하면 안 된다.

### 12.2 권장 공정 비교

모든 diffusion model에 동일한 scene split을 사용한다.

```text
train: data_train/objaverse_fixed32_coshadow_480/train.jsonl
val:   data_train/objaverse_fixed32_coshadow_480/val.jsonl
test:  data_train/objaverse_fixed32_coshadow_480/test.jsonl
```

PBR/LGI/GT에서 cache를 쓸 때 `dataset_metadata_path`만 바꾸면 안 된다.
`vae_latent_cache_specs`의 왼쪽 metadata도 같은 split manifest로 바꿔야 실제 dataset
row source가 일치한다.

권장 통일 항목:

- base checkpoint 동일
- scene split 동일
- train optimizer update 수 동일
- effective global batch 동일, 예: 24
- LR/WD 동일
- LoRA rank/target 동일
- CFG/dropout policy 기록
- inference seed/sampler/steps/CFG 동일
- 최소 3 random seeds

32,277 train rows, batch 1, accumulation 24, 10 epochs면 약 13,450 updates다.
epoch보다 optimizer update 수를 주 비교 기준으로 삼는 편이 정확하다.

### 12.3 추천 ablation 순서

```text
A. fixed32 RGB latent-only baseline
B. PBR joint depth+normal
C. LGI condition
D. CoShadow predicted bbox, layout only
E. CoShadow predicted bbox + mask auxiliary
F. CoShadow GT bbox oracle
G. GT direct/shadow dense-mask oracle
```

PBR은 추가로:

```text
B1. joint target inference
B2. GT depth/normal condition oracle
B3. depth-only
B4. normal-only
B5. source-drop probability 0 vs 0.12
```

LGI는:

```text
C1. LGI 3 channels only
C2. LGI + validity
C3. light only vs light+LGI
C4. spatial dropout 0 vs 0.1
```

### 12.4 평가 metric

전체 RGB:

- PSNR
- SSIM
- LPIPS
- 가능하면 color/brightness error

영역별:

- object mask 내부 RGB metric
- direct-lit 영역 metric
- cast-shadow 영역 metric
- non-shadow receiver 영역 metric

CoShadow:

- bbox IoU
- shadow presence accuracy와 precision/recall/F1
- shadow mask IoU/precision/recall
- empty-shadow false-positive rate
- predicted-vs-GT box downstream RGB gap

PBR:

- decoded depth error는 encoding을 먼저 역변환한 뒤 측정
- normal angular error
- joint-target와 GT-condition을 별도 표로 보고

LGI 자체는 condition이므로 reconstruction metric이 아니라 RGB/shadow 개선과
condition corruption/missing ablation을 본다.

GT mask 결과는 반드시 `oracle upper bound`로 분리한다.

## 13. 별도 물리-task CNN baseline

Wan 네 실험과 별도로 `model/physical_tasks/`에 세 개의 작은 CNN baseline이 있다.
이들은 TokenLight LoRA와 공동 optimizer로 학습하는 모듈이 아니라 독립 single-GPU
trainer다.

관련 manifest:

```text
data_train/physical_tasks_single_light/metadata.jsonl
  7,832 rows = train 7,354 + val 478

data_train/objaverse_fixed32_physical/metadata.jsonl
  35,614 rows = train 33,934 + val 1,680
```

split은 scene 단위다.

### 13.1 LightNet

목적: paired source/target RGB에서 target single-light parameter를 역추정한다.

입력:

```text
[source RGB, target RGB] = 6 channels
```

PBR image를 읽지 않는다. encoder는 residual/downsample CNN + global pool이고 10개 raw
value를 다음 의미로 변환한다.

```text
normalized direction (3)
log distance (1)
log intensity (1)
log radius (1)
sigmoid RGB color (3)
sigmoid ambient scale * 1.5 (1)
```

parameter 수: 6,779,306.

loss:

```text
direction cosine
+ smooth-L1 log distance
+ 0.5 * position smooth-L1
+ 0.5 * log intensity smooth-L1
+ 0.25 * log radius smooth-L1
+ 0.5 * color L1
+ 0.25 * ambient scale smooth-L1
```

중요: target RGB를 입력으로 사용하므로 일반 relighting inference의 source-only light
predictor가 아니다. inverse-light evaluation이나 teacher 용도다.

현재 output:

```text
outputs/physical_tasks/lightnet_single_light/
```

`best.pt`, `last.pt`, epoch checkpoint, config, metrics CSV가 존재한다. 저장된 run config는
480×480, batch 32, 50 epochs, LR 2e-4, WD 1e-4, seed 480이다. `last.pt`는 epoch 49다.

### 13.2 VisibilityNet

목적: object 내부의 front-facing/unoccluded direct-lit geometry mask를 예측한다.

입력은 15채널이다.

```text
source RGB                  3
reconstructed 3D point     3
world/canonical normal     3
pixel-to-light direction   3
log distance               1
N dot L                    1
object mask                1
                           --
                           15
```

metric depth, camera FOV/location/rotation으로 3D point를 재구성하고 recorded point light와
방향/거리를 계산한다. U-Net형 encoder-decoder가 1-channel logits를 출력한다.

parameter 수: 11,111,169.

loss domain은 object mask 내부다.

```text
L = balanced BCE + Dice
```

현재 output:

```text
outputs/physical_tasks/visibilitynet_fixed32/
```

저장된 run config는 480×480, batch 32, 최대 50 epochs, LR 2e-4다. 현재 `last.pt`는
epoch 42이며 `best.pt`가 존재한다. 즉 config의 50 epochs를 모두 완주했다고 단정하면
안 된다.

### 13.3 ShadowNet

목적: object가 receiver에 만드는 cleaned cast-shadow mask를 예측한다.

VisibilityNet 15채널 입력에 receiver mask 1채널을 더해 총 16채널이다.

```text
VisibilityNet features + receiver mask
```

target shadow mask는 입력이 아니다. U-Net형 network가 1-channel logits를 출력한다.

parameter 수: 11,111,489.

loss domain은 receiver mask 내부다.

```text
L = balanced BCE + Dice
```

현재 output:

```text
outputs/physical_tasks/shadownet_fixed32_clean/
```

저장된 run config는 480×480, batch 4, 최대 50 epochs, LR 2e-4다. 현재 `last.pt`는
epoch 45이며 `best.pt`가 존재한다.

### 13.4 세 CNN baseline의 runtime 제한

이 trainer들은 현재:

- 수동 `.to(cuda)`
- FP16 autocast + GradScaler
- 일반 DataLoader
- 직접 `torch.save`

를 사용한다. Accelerate distributed sampler, ZeRO-3, multi-rank metric gather, gradient
accumulation 공통 runtime에 연결돼 있지 않다. 작은 6.8M~11.1M CNN이라 parameter
sharding 이득도 작지만, “ZeRO-3 지원 모델”로 표기하면 안 된다.

또 `balanced_bce`의 positive/negative ratio가 batch-local aggregate이므로 multi-GPU로
단순 이식하면 single과 objective가 달라질 수 있다. DDP 이식 시 per-sample reduction
또는 global numerator/denominator 설계가 필요하다.

실행 예:

```bash
CUDA_VISIBLE_DEVICES=0 python -m model.physical_tasks.train_lightnet \
  --manifest data_train/physical_tasks_single_light/metadata.jsonl \
  --output-dir outputs/physical_tasks/lightnet_single_light \
  --image-size 480 --batch-size 16 --epochs 50

CUDA_VISIBLE_DEVICES=0 python -m model.physical_tasks.train_visibilitynet \
  --manifest data_train/objaverse_fixed32_physical/metadata.jsonl \
  --output-dir outputs/physical_tasks/visibilitynet_fixed32 \
  --image-size 480 --batch-size 4 --epochs 50

CUDA_VISIBLE_DEVICES=0 python -m model.physical_tasks.train_shadownet \
  --manifest data_train/objaverse_fixed32_physical/metadata.jsonl \
  --output-dir outputs/physical_tasks/shadownet_fixed32 \
  --image-size 480 --batch-size 4 --epochs 50
```

## 14. 기존 RGB/PBR/decoder run과 현재 safe 구현의 관계

### 14.1 기존 Zero-3 RGB/PBR artifacts

현재 다음 output이 존재한다.

```text
outputs/train/rgb_480_cache_allsets_lora128_zero3_b120
outputs/train/pbr_480_cache_allsets_lora128_zero3_b120
```

RGB run에는 `step-8000/16000/24000/32000.safetensors`와 full-step-32000 state가 있고,
PBR run에는 `step-16000/24000/32000.safetensors`가 있다.

resolved runtime상:

- mixed precision BF16
- 4-GPU로 추정되는 global batch 120
- RGB: microbatch 30 × 4
- PBR: microbatch 24 × 5로 기록된 global batch 120 계열
- LoRA rank 128

그러나 이 PBR output은 새 `train_tokenlight_pbr_safe.py`의 70/18/12 depth-normal safe
구현으로 학습한 결과가 아니다. legacy PBR artifact이며 새 safe 구현의 성공 증거로
사용하면 안 된다.

기존 `model/train_tokenlight_pbr.py`는 현재 import 시 다음 오류가 확인된다.

```text
ImportError: cannot import name '_constructor_resume_checkpoint'
             from model.train
```

따라서 앞으로 depth-normal PBR은 새 safe entrypoint를 사용한다.

### 14.2 2026-08-07 두 decoder run의 실제 차이

질문에 등장했던 두 run:

```text
tokenlight_480_rgb_decoder_w_shadowinput_maskloss_20260807_074452
tokenlight_480_rgb_decoder_w_shadowinput_maskloss_20260807_074746
```

resolved config 차이:

| 항목 | 074452 | 074746 |
|---|---:|---:|
| batch | 8 | 8 |
| accumulation | 1 | 5 |
| mask input token | true | true |
| RGB decoder outer weight | 0.0 | 1.0 |
| full decoder weight | 0.0 | 0.0 |
| shadow region weight | 0.0 | 0.1 |
| direct region weight | 0.0 | 0.1 |
| decoder point loss | MSE | MSE |
| runtime mixed precision | no | no |
| saved epochs | 0~9 | 0~7 |

즉 이름은 비슷하지만 074452는 실제로 decoder loss가 꺼진 latent-only + shadow-mask
input run이고, 074746은 shadow/direct region decoder loss가 켜진 run이다. effective
batch도 8 vs 40으로 다르므로 decoder 하나만의 완전 통제 실험도 아니다.

074452에는 full-step-44510 model/optimizer/scheduler/RNG state가 있고, 074746은 epoch
portable weights만 확인된다.

두 run 모두 direct `python` single 경로였고 runtime `mixed_precision=no`인데 trainable
parameter가 BF16 AdamW 경로였을 가능성이 있어 optimizer precision도 현재 safe
정책과 다르다.

### 14.3 2026-08-11 safe current-state 실험 흔적

다음 두 output에는 resolved safe 설정과 TensorBoard log가 있으나 portable checkpoint는
없다.

```text
..._20260811_074309: batch 5 × accumulation 8
..._20260811_074543: batch 10 × accumulation 4
```

둘 다 effective batch 40이며 resolved 설정은 대략:

- mask token false
- current-t decoder outer weight 1
- full loss 1
- shadow/direct region 0
- L1
- BF16 autocast

짧은 startup/메모리 실험으로 보이며 완성된 학습 결과로 보면 안 된다.

### 14.4 기존 decoder config 두 개

```text
configs/train_480/rgb_decoder.json
configs/train_480/rgb_decoder_w_mask.json
```

현재 내용은 기능상 거의 같고 batch/accumulation만 다르다.

```text
rgb_decoder.json:        batch 5 × accumulation 8 = 40
rgb_decoder_w_mask.json: batch 8 × accumulation 5 = 40
```

둘 다 shadow mask condition, full decoder weight 0, shadow/direct region 각각 0.1,
MSE를 명시한다. 새 current-state L1 safe objective를 즉시 사용하려면 config/CLI override를
명시해야 하며 이 두 JSON 이름만 보고 safe objective라고 판단하면 안 된다.

## 15. 현재 산출물 상태

### 15.1 생성 완료

- 네 신규 trainer/model/config source
- 공통 physics runtime
- PBR safe inference wrapper
- CoShadow 전체 35,614-row metadata와 split
- CoShadow unit tests
- LGI/GT model-free preflight
- physical-task manifests
- 기존 LightNet/VisibilityNet/ShadowNet checkpoints
- 이 종합 문서와 물리조건부 설계 문서

### 15.2 아직 생성되지 않음

현재 다음 기본 output directory에는 장시간 학습 checkpoint가 없다.

```text
outputs/train/tokenlight_480_fixed32_pbr_depth_normal_safe
outputs/train/coshadow_box_fixed32
outputs/train/tokenlight_480_coshadow
outputs/train/tokenlight_480_rgb_spatial_lgi
outputs/train/tokenlight_480_rgb_spatial_gt_masks
```

즉 다음은 아직 없다.

- 새 safe PBR 장시간 학습 결과
- CoShadow box predictor 학습 결과와 bbox visualization
- CoShadow diffusion 결과
- LGI 학습 결과
- GT-mask oracle 학습 결과
- CoShadow/LGI/GT 전용 inference 생성 결과

### 15.3 inference 지원 matrix

| 경로 | training | model-free preflight | 전용 inference |
|---|---|---|---|
| PBR safe | 구현 | 구현 | 구현 (`infer_manifest_pbr_safe.py`) |
| CoShadow box | 구현 | 구현 | predictor loader는 있으나 시각화 CLI 없음 |
| CoShadow diffusion | 구현 | 구현 | 미구현 |
| LGI | 구현 | 구현 | 미구현 |
| GT masks | 구현 | 구현 | 미구현 |
| decoder safe RGB | 구현 | config validation | 기존 RGB inference 재사용 가능하나 objective 검증 필요 |

## 16. 실행 및 검증 기록

사용자 요청에 따라 실행성 검사는 모두 다음 컨테이너 안에서만 수행했다.

```text
container ID: f8c4e2e879bd
container name: jaeho_relighting
working dir: /workspace
```

host에서 직접 Python 학습/검증을 실행하지 않았다.

### 16.1 코드/config

- 신규 Python 전체 `py_compile` 통과
- PBR/LGI/GT/CoShadow entrypoint `--help` 및 config parse 통과
- JSON config 5개 parse 통과
- 기존 canonical `model/train.py` diff 없음
- 기존 `model/tokenlight_wan_pbr.py` diff 없음

### 16.2 PBR preflight

실제 출력 요약:

```text
reject filter: 48,835 → 35,614
samples checked: 2
depth: 480×480 RGB full decode
normal: 480×480 RGB full decode
RGB cached latent: [48,1,30,30]
status: ok
```

### 16.3 LGI preflight

초기 sample `scene_000001/position_00`에서:

```text
shape: [4,480,480]
ordering error: 0
NaN/Inf: 0
channel means:
  min       0.7329043
  max       1.5006050
  nearest   0.7712404
  validity  0.9979167
```

이 값은 한 sample의 파일 통계이며 데이터 전체 normalization 상수가 아니다.

이후 coverage 재집계는 28,032 complete LGI directories, filtered metadata와 일치하는
20,485 rows로 갱신됐다. 기존 `docs/PHYSICS_CONDITIONED_TRAINING.md`의 “예시 한 개”
문구보다 이 문서의 2026-08-13 snapshot이 최신이다.

### 16.4 GT-mask preflight

검사 sample에서:

```text
condition shape: [2,480,480]
channel order: [direct_lit, cast_shadow]
object input: validation only
binary/relation errors: 0
```

여러 metadata offset 및 `position_002`를 포함하는 scene preflight가 통과했다.

### 16.5 CoShadow metadata/validation

- full metadata build: 35,614 output, failures 0
- scene-disjoint train/val/test 확인
- box dataset actual PNG sample load 통과
- diffusion metadata/bbox/light/PNG/RGB cache model-free validation 통과
- unit tests 8/8 통과

실행한 test:

```bash
python -m unittest tests/test_coshadow.py -v
```

검사 범위:

- empty/edge bbox와 half-open normalization
- half-bin round 방식
- deterministic scene split
- CoordConv box forward/loss
- box checkpoint schema roundtrip
- layout embedding
- batch 2 timestep promotion
- unsupported attention alignment fail-fast
- bbox metadata contract

### 16.6 검증이 의미하지 않는 것

위 결과는 다음을 증명하지 않는다.

- 5B Wan 장시간 학습이 수렴한다는 것
- 새 condition이 RGB quality를 개선한다는 것
- inference pipeline이 PBR 외 세 경로에서 완성됐다는 것
- LGI 단위/좌표계가 논문과 정확히 같다는 것
- depth/normal encoding이 metric physical prediction에 적합하다는 것

현재 검증은 code import, tensor contract, data decode, loss gradient path, metadata
consistency 단계다. 실제 연구 결론에는 short overfit, checkpoint inference, held-out
evaluation이 추가로 필요하다.

## 17. 권장 다음 작업 순서

1. **동일 fixed32 split manifest 준비**
   - PBR/LGI/GT용 train/val/test metadata와 cache spec을 CoShadow split과 맞춘다.
2. **LGI 나머지 15,129 rows 생성**
   - unit, camera convention, invalid sentinel을 sidecar schema로 저장한다.
3. **각 모델 16~64 sample overfit**
   - RGB latent-only baseline부터 condition이 실제로 loss/출력에 영향을 주는지 확인한다.
4. **CoShadow box predictor 먼저 학습/시각화**
   - GT/pred bbox overlay, empty-shadow confusion matrix, val IoU를 확인한다.
5. **CoShadow diffusion GT-box smoke 후 predicted-box 학습**
   - GT-box에서도 실패하면 layout injection을 먼저 수정한다.
6. **LGI/GT/CoShadow inference entrypoint 추가**
   - 학습 auxiliary module과 token/timestep semantics를 동일하게 복원한다.
7. **동일 optimizer updates/global batch로 본 학습**
8. **RGB 전체/영역별 metric 및 3 seeds 평가**
9. **필요할 때만 decoder objective를 opt-in ablation으로 추가**

## 18. 재현 체크리스트

학습을 시작하기 전에 run snapshot에서 다음을 반드시 확인한다.

- [ ] 올바른 entrypoint인가?
- [ ] metadata와 `vae_latent_cache_specs` 왼쪽 manifest가 같은 split인가?
- [ ] reject list가 적용됐는가?
- [ ] condition channel order가 checkpoint schema와 같은가?
- [ ] source/light/spatial condition은 `t=0`, target은 sampled `t`인가?
- [ ] decoder loss가 의도대로 0인가?
- [ ] single이면 trainable parameter와 Adam state가 FP32인가?
- [ ] ZeRO-3이면 실제 DeepSpeed plugin으로 launch했는가?
- [ ] `micro × accumulation × world_size`가 비교 실험과 같은가?
- [ ] CoShadow predicted mode에서 box checkpoint가 실제로 load됐는가?
- [ ] PBR inference가 safe wrapper를 사용하는가?
- [ ] LGI/GT/CoShadow inference가 auxiliary module까지 load하는가?
- [ ] oracle condition 결과를 일반 inference 결과와 분리했는가?

## 19. 한 문장씩 정리

- **PBR**: depth와 normal을 RGB와 공동 target/condition으로 다루는 UniRelight식
  latent joint-stream adaptation이다.
- **CoShadow**: light-conditioned bbox predictor의 coarse shadow layout을 네 learned token으로
  Wan에 넣고 RGB velocity와 shadow mask를 학습하는 2단계 adaptation이다.
- **LGI**: `[min,max,nearest-to-zero]` light-geometry map과 local validity를 dense spatial
  prefix로 넣는 latent-only adaptation이다.
- **GT mask**: 정답 direct-lit/cast-shadow 두 채널을 condition으로 주는 diagnostic upper
  bound다.
- **decoder safe**: 최종 trajectory loss가 아니라 sampled current-flow decoder metric을
  실험하는 별도 opt-in 경로다.
- **physical CNNs**: LightNet/VisibilityNet/ShadowNet은 독립 single-GPU teacher/baseline이며
  Wan 네 실험과 joint train하지 않는다.
