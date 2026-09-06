# TokenLight에 MoGe-3 spatial geometry condition을 넣는 설계

작성일: 2026-08-20  
대상: 현재 fixed32/Objaverse TokenLight RGB relighting 모델

## 0. 결론

제안의 큰 방향은 맞다. MoGe-3의 point map을 하나의 global token으로 압축하지 말고, source image와 같은 pixel 위치를 유지하는 dense spatial condition으로 넣어야 한다.

현재 모델에 가장 적합한 최종 형태는 다음이다.

```text
Source RGB ─ VAE ─ source tokens S ───────────────────────────┐
                                                             │
Source RGB ─ frozen MoGe-3 ─ P, N, Z, valid                  │
                              │                              │
                              ├─ static map Gscene ─ Escene ─┤ residual add
Target light attrs ─ L ───────┴─ relative map Glight ─ Elight┤
                                                             ▼
                                         S' = S + gs Es + gl El

Noisy target RGB tokens + 기존 scalar Lightoken ────────── Wan DiT
                                                             │
                                                             ▼
                                                       Relighted RGB
```

첫 구현의 권장 사양은 아래와 같다.

- MoGe-3는 frozen, scene당 한 번만 precompute한다.
- 기존 scalar Lightoken은 그대로 유지한다.
- static geometry는 8채널 `point(3)+normal(3)+log-depth(1)+valid(1)`로 시작한다.
- dynamic geometry는 light당 7채널 `direction(3)+log-distance(1)+signed N·L(1)+bounded attenuation(1)+slot-valid(1)`로 시작한다.
- dense XYZ에는 처음부터 고주파 Fourier encoding을 넣지 않는다.
- 두 geometry encoder의 출력을 source token grid에 residual로 더한다. 새 spatial token 225개를 append하지 않는다.
- static geometry는 CFG negative branch에도 유지하고, dynamic light map만 scalar light token과 같은 dropout mask로 제거한다.
- `RGB baseline → GT geometry → GT geometry+relative light map → MoGe oracle-alignment → deployable MoGe alignment` 순서로 검증한다.

가장 중요한 것은 encoder보다 좌표계다. 현재 제안의 축 변환은 맞지만 camera 원점과 scale 정렬이 빠져 있다. 이를 해결하지 않고 `L-P`를 계산하면 geometry condition이 물리 정보가 아니라 잘못된 위치 신호가 된다.

---

## 1. 제안에서 맞는 부분과 수정해야 할 부분

### 맞는 부분

1. Point map은 global token보다 spatial map이 적합하다.
2. Static scene geometry와 light-dependent geometry를 분리해야 한다.
3. Point와 light를 결합한 `L-P`, direction, distance, `N·L`을 명시적으로 제공하는 것이 좋다.
4. 기존 scalar Lightoken은 전역 광원 속성 전달용으로 계속 필요하다.
5. 동일 scene의 여러 light target이 같은 geometry를 공유하도록 해야 한다.
6. MoGe는 학습 loop 안에서 실행하지 않고 scene/source별로 cache해야 한다.

### 반드시 수정할 부분

1. `P_TL = A P_MoGe`만으로는 부족하다. 이 값은 camera-origin 기준 벡터이며, 현재 TokenLight canonical camera 원점은 `[0,-3.5,0]`이다.
2. 올바른 축 행렬은 아래의 `A`다. 제안문에 렌더링된 행렬은 코드의 `[x,z,-y]`와 부호가 달라 보인다.
3. MoGe의 metric unit과 Blender normalized scene unit은 같지 않다. Blender metadata의 `canonical_scale`을 MoGe point에 그대로 적용하면 안 된다.
4. MoGe `depth`는 Euclidean range가 아니라 OpenCV camera-space의 `z`다. Light distance는 반드시 `||L-P||`로 계산해야 한다.
5. MoGe-3 `infer()`가 반환하는 `points/depth`에는 내부 `metric_scale`이 이미 적용된다. cache에 별도 `metric_scale` 출력이 있다고 가정하면 안 된다.
6. 현재 저장소에는 이미 전용 CNN spatial encoder와 spatial-prefix trainer가 있다. 따라서 natural-RGB용 VAE에 XYZ를 넣는 PBR 우회보다 이 encoder를 직접 재사용하는 편이 빠르고 정확하다.
7. 제안의 gate 예시는 `gate_logit`을 선언했지만 실제 forward에서 곱하지 않았다. 최종 구현에는 gate 적용이 반드시 들어가야 한다.

---

## 2. 현재 코드와 데이터가 실제로 사용하는 좌표계

### 2.1 네 개의 좌표계를 구분한다

| 이름 | 원점 | 축 | 단위/용도 |
|---|---|---|---|
| Image pixel | image top-left | `u=right`, `v=down` | 2D spatial 위치 |
| MoGe/OpenCV camera | camera optical center | `+X=right`, `+Y=down`, `+Z=forward` | MoGe point/normal/depth |
| TokenLight canonical | scene canonical center | `+x=right`, `+y=forward`, `+z=up` | 현재 light attrs |
| Blender world | Blender scene origin | metadata의 similarity transform에 따라 달라짐 | renderer 내부 |

현재 fixed32 physical reconstruction도 camera ray를 다음처럼 만든다.

```python
ray_can = [screen_x * tan(fov / 2), 1, -screen_y * tan(fov / 2)]
```

즉 현재 TokenLight canonical 축은 `right-forward-up`이다. 근거 코드는 `model/physical_tasks/data.py:105-131`이다.

### 2.2 MoGe/OpenCV에서 TokenLight canonical 방향으로 회전

MoGe 공식 output은 point와 normal을 OpenCV camera coordinate로 정의한다([공식 README example](https://github.com/microsoft/MoGe#-minimal-code-example), [v3 API](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v3.py#L197-L232)). 정확한 축 변환은 다음과 같다.

\[
A_{cv\rightarrow can}=
\begin{bmatrix}
1&0&0\\
0&0&1\\
0&-1&0
\end{bmatrix}
\]

따라서 camera-origin 기준 벡터는 다음처럼 변한다.

\[
q_{can}=A P_{cv}=(X_{cv},Z_{cv},-Y_{cv})
\]

```python
points_axis_can = torch.stack(
    [
        points_cv[:, 0],    # right  -> canonical +x
        points_cv[:, 2],    # forward -> canonical +y
        -points_cv[:, 1],   # down -> canonical -z
    ],
    dim=1,
)

normals_can = torch.stack(
    [normals_cv[:, 0], normals_cv[:, 2], -normals_cv[:, 1]],
    dim=1,
)
normals_can = torch.nn.functional.normalize(normals_can, dim=1, eps=1e-6)
```

`det(A)=+1`인 순수 회전이므로 normal에도 같은 행렬을 적용한다. Translation과 uniform scale은 normal에 적용하지 않는다.

### 2.3 빠진 camera 원점

현재 fixed32 계열의 canonical camera 위치는 전 scene에서 다음과 같다.

```text
O_can = [0, -3.5, 0]
FOV_x = 39.6 degrees
```

따라서 canonical absolute point는 단순히 `A P_cv`가 아니라 다음이다.

\[
P_{can}=O_{can}+\alpha_{moge} A P_{cv}
\]

여기서 `alpha_moge`는 `canonical unit / MoGe unit`이다. `O_can`을 더하지 않고 canonical light에서 point를 빼면 forward 축 위치가 약 3.5만큼 틀어진다.

### 2.4 현재 light token은 canonical position이다

`scripts/build_final_objaverse_light_mask_infer_manifest.py:151-166`은 renderer metadata의 `light.canonical_position`을 읽어 `attrs_json.lights[0].x/y/z`에 넣는다. 즉 기존 Lightoken의 `x,y,z`는 Blender world가 아니다.

현재 grid는 다음 범위다.

```text
x ∈ {-0.75, -0.25, 0.25, 0.75}
y ∈ {-0.75, -0.25, 0.25, 0.75}
z ∈ { 0.25,  0.75}
```

scene metadata의 renderer transform은 다음이다.

\[
p_{world}=C_t+sR(p_{can}-C)
\]

역변환은 다음이다.

\[
p_{can}=C+\frac{1}{s}R^T(p_{world}-C_t)
\]

Renderer world point를 사용할 때는 이 식이 정확하다. MoGe point에는 renderer의 `s`를 그대로 쓰면 안 된다.

---

## 3. 권장 좌표 처리: dense branch는 MoGe camera frame을 유지한다

수식과 디버깅은 canonical frame으로 해도 되지만, 실제 dense branch는 MoGe native OpenCV camera frame을 유지하는 편이 실수가 적다. Point와 normal을 이동시키는 대신 canonical light를 camera frame으로 바꾼다.

아래의 단순한 camera-frame 식은 alignment translation을 canonical camera 원점으로 고정한 restricted mode, 즉 `P_can=O_can+alpha_moge A P_cv`를 전제로 한다.

먼저 MoGe point를 canonical 길이 단위로만 scale한다.

\[
\hat P_{cv}=\alpha_{moge} P_{cv}
\]

Canonical light의 camera-relative vector를 OpenCV 축으로 회전한다.

\[
\hat L_{cv}=A^T(L_{can}-O_{can})
\]

현재 `O_can=[0,-3.5,0]`이므로 이를 풀어 쓰면 다음과 같다.

```python
light_cv_can = torch.stack(
    [
        light_can[..., 0],          # x right
        -light_can[..., 2],         # y down
        light_can[..., 1] + 3.5,    # z forward from camera
    ],
    dim=-1,
)
```

그다음 물리 relation을 동일한 camera frame에서 계산한다.

```python
point_cv_can = alpha_moge[:, None, None, None] * points_cv
vector = light_cv_can[..., None, None] - point_cv_can[:, None]
distance = torch.linalg.vector_norm(vector, dim=2, keepdim=True).clamp_min(1e-6)
direction = vector / distance
ndotl = (normals_cv[:, None] * direction).sum(dim=2, keepdim=True)
```

이 방식의 장점은 다음과 같다.

- source image의 `right/down` 방향과 dense point/normal map이 그대로 정렬된다.
- canonical camera translation을 빠뜨릴 가능성이 작다.
- Point, normal을 다시 회전하면서 생길 구현 혼동이 줄어든다.
- Direction, distance, `N·L`은 직교 회전에 대해 불변이므로 canonical frame 계산과 물리적으로 동일하다.
- 기존 scalar Lightoken에는 지금과 같은 canonical `x,y,z`를 계속 넣을 수 있다.

Oracle 진단에서 free translation `t`까지 fit하여 `P_can=alpha_moge A P_cv+t`를 쓴다면 point의 camera-relative 좌표에도 translation correction이 필요하다.

\[
P^{rel}_{cv}
=A^T(P_{can}-O_{can})
=\alpha_{moge}P_{cv}+A^T(t-O_{can})
\]

\[
V_{cv}=A^T(L_{can}-O_{can})-P^{rel}_{cv}
\]

이 보정을 빠뜨리면 free-translation oracle의 distance와 `N·L`이 틀린다. Free `t`는 diagnostic에만 두고, 구현은 canonical frame에서 `L_can-(alpha_moge A P_cv+t)`를 직접 계산하는 편이 더 안전하다.

Canonical map 자체가 필요하면 다음으로 바꾸면 된다.

\[
P_{can}=O_{can}+A\hat P_{cv},\qquad N_{can}=A N_{cv}
\]

두 구현의 `distance`와 `N·L`이 수치적으로 같아야 한다. 이것을 unit test로 둔다.

---

## 4. 가장 어려운 부분: scale/gauge 정렬

### 4.1 Renderer scale과 MoGe metric scale은 다르다

Renderer camera-space point가 Blender world unit으로 정확하게 주어졌다면 다음을 쓸 수 있다.

\[
\alpha_{render}=1/s_{world}
\]

이는 camera-origin 기준 vector가 Blender world-length 단위일 때의 scale이다. Absolute world `Position` pass라면 `1/s`만 곱하지 말고 §2.4의 full inverse `C+(1/s)R^T(P_world-C_t)`를 사용한다.

그러나 MoGe는 open-domain metric scale을 추정하며 Blender의 임의 object normalization unit을 알지 못한다. 따라서 다음은 틀린 구현이다.

```python
# 틀림: Blender unit과 MoGe metric unit을 동일시함
point_cv_can = points_moge_cv / meta["camera"]["canonical_scale"]
```

Objaverse의 stylized/close-up render는 MoGe metric scale이 특히 불안정할 수 있으므로 별도 calibration이 필요하다. 공식 app도 일반적인 indoor/street 영역 밖의 stylized·close-up 입력에서는 metric scale 신뢰도가 낮아질 수 있다고 [경고한다](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/scripts/app.py#L253-L256).

### 4.2 연구를 세 개의 alignment mode로 나눈다

| mode | 방법 | 목적 | 실제 inference 가능 여부 |
|---|---|---|---|
| `gt_canonical` | Blender float point를 metadata로 정확히 canonical 변환 | architecture 상한 검증 | synthetic에서만 가능 |
| `moge_oracle_scale` | 같은 pixel의 GT point로 `alpha_moge_oracle`과 shift를 robust fit | MoGe shape 오차와 gauge 오차 분리 | oracle, 배포 불가 |
| `moge_deployable` | source/object mask와 고정 rig 또는 scale head만 사용 | 실제 사용 조건 | 가능 |

이 세 결과를 섞어서 보고하면 안 된다. 특히 GT-fit을 test scene에도 적용한 결과는 `MoGe` 결과가 아니라 `oracle-aligned MoGe` 결과다.

### 4.3 첫 oracle은 float Position pass를 권장한다

현재 random2 데이터에는 8-bit percentile depth PNG와 8-bit normal PNG가 있다. 기존 physical loader는 depth PNG를 min/max metadata로 복원하지만 다음 오차가 남는다.

- 8-bit quantization
- 1/99 percentile clipping
- object boundary interpolation

구조 검증용 GT oracle은 가능하면 Blender의 float32 EXR `Position` pass 또는 unclipped float depth를 scene당 한 번 다시 저장한다. 기존 PNG는 빠른 smoke test에는 쓸 수 있지만 정확한 GT라고 부르지 않는다.

또 현재 `model/physical_tasks/data.py:105-123`은 복원한 PBR depth를 unit camera ray에 곱하므로 이 PNG를 **camera-ray range**로 취급한다. 반면 MoGe `depth`는 camera-space `Z`다. 두 depth scalar의 ratio를 직접 scale로 사용하지 말고, 각각 3D point로 복원한 뒤 point-to-point scale을 fit한다. 새 cache schema에도 `ray_range`인지 `camera_z`인지 명시한다.

현재 PBR normal PNG는 world-space다. GT를 camera-frame dense branch에 넣으려면 metadata rotation `R`과 축 행렬 `A`를 사용한다.

\[
N_{can}=R^TN_{world},\qquad
N_{cv}=A^TR^TN_{world}
\]

World normal PNG를 MoGe camera normal처럼 그대로 사용하면 `N·L`의 frame이 어긋난다.

### 4.4 Oracle scale fit

MoGe camera point를 축만 맞춘 `q=A P_cv`라 하고, 대응 GT canonical point를 `P_gt`라 하자. 가장 단순한 oracle은 positive scalar와 translation을 robust fit한다.

\[
\min_{\alpha_{\mathrm{moge,oracle}}>0,t}\sum_{i\in M}
\rho(\|\alpha_{\mathrm{moge,oracle}} q_i+t-P^{GT}_{i}\|_2)
\]

현재 fixed camera rig의 원점을 강제하려면 `t=O_can`으로 고정하고 `alpha_moge_oracle`만 fit할 수 있다. 먼저 이 restricted fit을 사용하고, residual이 큰 경우에만 free translation fit을 진단용으로 본다. 임의 회전까지 허용하면 축 버그를 similarity fit이 숨길 수 있으므로 첫 실험에서는 회전을 고정한다.

Free `t`를 실제 feature 생성에 사용할 때는 §3의 translation correction을 적용한다. `point_cv_can=alpha_moge*points_cv`만 사용하는 camera-frame fast path는 `t=O_can` mode 전용이다.

Fit support는 `object_mask ∩ moge_valid ∩ gt_valid`로 두고, fit된 transform은 receiver/floor/wall을 포함한 모든 valid point에 동일하게 적용한다.

### 4.5 현재 고정 rig에서 가능한 deployable 초기값

현재 camera-object nominal distance가 3.5이므로 가장 단순한 초기값은 다음이다.

\[
\alpha_{\mathrm{moge,init}}=
\frac{3.5}
{\operatorname{median}_{q\in M_{obj}} Z_{MoGe}(q)}
\]

이는 visible front surface의 median을 object center로 보는 근사이므로 bias가 있다. 최종안으로 고정하기보다 다음처럼 사용한다.

1. Synthetic train scene에서 `alpha_moge_oracle`을 구한다.
2. `alpha_moge_init/alpha_moge_oracle`의 dataset 분포를 측정한다.
3. dataset-wide correction 또는 작은 scale-calibration head를 학습한다.
4. Unseen scene에서는 source RGB, object mask, MoGe depth quantile, FOV만으로 `log alpha_moge_pred`를 예측한다.

Scale head의 teacher는 GT-fit scale이고, relighting DiT와 분리해서 먼저 검증하는 편이 좋다.

### 4.6 Object-centric normalization의 oracle 정의와 배포 조건

Camera rig가 고정되지 않거나 metric scale을 포기하고 완전히 dimensionless한 model을 만들려면 object-centric normalization을 사용할 수 있다. 다만 point만 normalize하면 안 된다.

GT/canonical branch에서:

\[
c_g=\operatorname{median}_{M_{obj}} P_{gt},\qquad
r_g=Q_{0.95}(\|P_{gt}-c_g\|)
\]

\[
P'_g=(P_{gt}-c_g)/r_g,\quad
L'=(L_{can}-c_g)/r_g,\quad
O'_g=(O_{can}-c_g)/r_g
\]

MoGe branch에서 같은 pixel support로:

\[
c_m=\operatorname{median}_{M_{obj}} (A P_{moge}),\qquad
r_m=Q_{0.95}(\|A P_{moge}-c_m\|)
\]

\[
P'_m=(A P_{moge}-c_m)/r_m
\]

두 point cloud가 전역 similarity 관계이고 동일 support를 사용한다면 `P'_m≈P'_g`가 된다. 이때 light는 GT/canonical 쪽에서 정의된 `L'`을 사용한다. 하지만 이 식은 `c_g,r_g`를 GT point에서 구하므로 그대로는 **oracle object-centric normalization**이지 배포 가능한 방법이 아니다.

배포 가능하게 만들려면 다음 중 하나가 필요하다.

- Canonical center/radius가 source와 함께 metadata로 제공된다.
- User light position 자체를 미리 정의한 object-normalized frame으로 입력한다.
- Source/MoGe만으로 canonical↔MoGe gauge를 예측하는 calibration head를 학습한다.
- 현재 fixed rig처럼 camera origin은 고정하고 `alpha_moge_pred` 하나만 예측한다.

현재 데이터와 기존 Lightoken checkpoint에는 마지막 방법이 가장 안전하다.

이 mode를 최종 채택하면 dynamic map만 `L'`을 쓰고 scalar Lightoken은 예전 `L_can`을 쓰는 식으로 좌표 계약을 암묵적으로 섞지 않는 편이 좋다. Full retraining에서는 scalar Lightoken도 `L'`로 바꾸고, 기존 checkpoint를 warm-start해야 한다면 `L_can`과 `L'` 중 무엇을 어느 branch에 넣는지 schema에 명시한 별도 ablation으로 둔다.

주의할 점은 다음과 같다.

- 통계 mask는 MoGe valid 전체가 아니라 `object mask ∩ MoGe valid`여야 한다. 전체 valid를 쓰면 floor/wall이 center를 지배한다.
- Transform을 추정할 때만 object를 사용하고, 추정한 transform은 전체 visible scene/receiver point에 적용한다.
- XYZ 각 축을 따로 standardize하면 Euclidean direction과 distance가 깨진다. 하나의 scalar radius만 사용한다.
- Visible point의 median은 실제 watertight object center가 아니므로 이 방식도 근사다.
- 이 방법은 object mask를 필요로 한다. 실제 inference에 object mask가 없다면 source-only segmentation을 함께 제공하거나, object mask 없이 동작하도록 학습한 scale/gauge head가 필요하다. Target RGB나 GT geometry로 scale을 정하면 inference leakage다.
- Object-centric transform을 쓰면 camera distance `d0`, light radius, 모든 distance도 같은 radius로 나눠야 한다. Calibrated mode의 `d0=3.5`를 변환 없이 재사용하면 단위가 섞인다.
- 현재 fixed canonical rig에서는 임의 translation normalization보다 camera 원점을 보존하는 scale-calibration 방식이 1순위다.

---

## 5. Coordinate encoding 설계

여기서 encoding을 세 종류로 분리해야 한다.

1. **2D pixel 위치 encoding**: Wan의 기존 RoPE가 이미 처리한다.
2. **3D scene coordinate encoding**: geometry CNN에 넣는 point/normal/depth 값이다.
3. **Global light coordinate encoding**: 기존 Lightoken Gaussian Fourier projection이다.

2D RoPE가 있으므로 dense point map에 image `u,v`를 무조건 반복해 넣을 필요는 없다. 더 중요한 것은 point와 light의 상대 관계다.

### 5.1 V1 static geometry: 8채널

권장 camera-frame 입력은 다음과 같다.

```text
scaled camera-space XYZ / d0       3
camera-space normal                3
relative log camera-Z              1
geometry validity                  1
                                  ---
                                    8
```

여기서 `d0=3.5`는 현재 canonical camera distance다.

```python
p = point_cv_can / 3.5
z = point_cv_can[:, 2:3].clamp_min(1e-6)
log_z = torch.log(z / 3.5)
```

`log_z`는 raw calibrated XYZ를 대체하지 않고 보조 채널로 둔다. Absolute scale을 보존해야 하므로 XYZ와 distance는 image별 z-score로 바꾸지 않는다. Channel normalization은 train set 전체에서 구한 고정 mean/std를 checkpoint buffer로 저장한다.

Invalid point는 다음 순서로 처리한다.

```python
# 아래는 points/normals=[B,3,H,W], moge_mask=[B,H,W]로 바꾼 뒤의 예다.
valid = moge_mask.bool() & torch.isfinite(points).all(dim=1) & (points[:, 2] > 0)
points = torch.where(valid[:, None], points, 0)
normals = torch.where(valid[:, None], normals, 0)
```

Zero가 실제 좌표인지 invalid인지 구분하기 위해 valid channel이 반드시 필요하다.

### 5.2 V2 static geometry: factorized coordinate 추가

MoGe-3는 내부 refinement에서 `(X/Z, Y/Z, log Z)`를 사용한다. 필요하면 V1에 ray coordinate 두 개를 추가한다. 이 factorization과 log-depth refinement는 [MoGe-3 §3.2](https://arxiv.org/html/2607.17967#S3.SS2)에 설명되어 있다.

MoGe/OpenCV frame에서는:

\[
u=X/Z,\qquad v=Y/Z,\qquad \zeta=\log Z
\]

TokenLight right-forward-up의 **absolute point** `P_can`을 쓸 때는 camera origin을 빼야 한다. `q=P_can-O_can`이라 두면:

\[
u=q_x/q_y=\frac{P_x}{P_y+3.5},\qquad
v=-q_z/q_y=-\frac{P_z}{P_y+3.5}
\]

그러면 static map은 10채널이 된다.

```text
XYZ 3 + normal 3 + ray(u,v) 2 + log-depth 1 + valid 1 = 10
```

다만 `u,v`는 camera intrinsics와 2D RoPE에 상당 부분 중복된다. 처음부터 넣지 말고 8채널 baseline 후 ablation한다.

### 5.3 V1 dynamic light geometry: light당 7채널

각 pixel과 light `k`에 대해 다음을 계산한다.

\[
V_k=L_k-P,\quad
r_k=\|V_k\|,\quad
D_k=V_k/(r_k+\epsilon)
\]

\[
c_k=N^T D_k
\]

권장 channel은 다음이다.

```text
pixel-to-light unit direction       3
log1p(distance / d0)                1
signed N dot L                      1
1 / (1 + (distance / d0)^2)         1
geometry-valid × light-slot-valid   1
                                    -
                                    7
```

초기에는 `N·L`을 ReLU로만 자르지 않고 signed 값으로 넣는다. Back-facing 여부도 유용한 단서다. 필요하면 `ReLU(N·L)`을 한 채널 더 넣는 ablation을 한다.

Raw inverse square는 근거리에서 폭발할 수 있으므로 bounded attenuation으로 시작한다. Area-light radius까지 반영하려면 후속 실험에서 다음과 같이 완화한다.

\[
a_k=\frac{1}{(r_k/d_0)^2+(d_k/d_0)^2+\epsilon}
\]

이 값은 train-set percentile로 clip하거나 log-compress한 뒤 넣는다.

### 5.4 Optional irradiance prior

Scalar Lightoken만으로도 color/intensity/radius를 전달할 수 있지만, model이 전역 scalar와 pixel distance를 곱하는 부담을 줄이려면 다음 RGB prior를 추가할 수 있다.

\[
E_k^{prior}=\lambda_k c_k^{rgb}
\frac{\max(N^TD_k,0)}{1+(r_k/d_0)^2}
\]

이는 renderer 정답이 아니라 local direct-light prior다. Cast visibility, indirect light, BRDF는 포함하지 않으므로 condition으로만 사용하고 reconstruction target으로 강제하지 않는다.

### 5.5 Dense XYZ Fourier encoding은 나중에 한다

현재 scalar Lightoken은 `model/lightoken_encoder.py:107-193`에서 각 scalar를 `sigma=5`, 512-feature random Gaussian Fourier로 encoding한다. 기존 checkpoint 호환을 위해 이는 유지한다.

하지만 같은 고주파 random encoding을 noisy MoGe XYZ에 그대로 적용하는 것은 첫 실험으로 권하지 않는다.

- Depth boundary/fly-point noise를 고주파로 증폭한다.
- Canonical range 밖 extrapolation이 불안정할 수 있다.
- 이미 relative direction/distance/`N·L`에 유용한 비선형성이 들어 있다.

필요하면 raw coordinate를 남긴 채 낮은 주파수의 deterministic encoding만 추가한다.

\[
\gamma(q)=
[q,\sin(\pi2^kq),\cos(\pi2^kq)]_{k=0,1,2}
\]

Direction에는 degree 2 이하 spherical harmonics, log-distance에는 작은 RBF bank를 후속 ablation으로 비교할 수 있다. V1은 raw physical channel만 사용한다.

### 5.6 Multi-light encoding

현재 config의 `max_lights=2`지만 fixed32 row는 실제로 single-light다. V1에서는 `[B,K,7,H,W]`로 만들고 동일한 per-light encoder를 공유한 뒤 masked sum하는 방식이 가장 깔끔하다.

\[
E_{light}=\sum_k m_k E_{one}(G_{light,k})
\]

이렇게 하면 light slot 순서를 바꿔도 dense condition이 동일하다. 단순히 14채널을 concatenate하면 기존 Lightoken slot 순서와 정확히 맞춰야 하며 permutation invariant하지 않다.

주의할 점은 기존 `LightokenEncoder(max_lights=2)`가 `light0_*`, `light1_*`에 서로 다른 projection을 사용하므로 dense branch만 합산해도 **전체 모델**은 아직 permutation invariant하지 않다는 것이다. 실제 two-light 학습 전에는 attrs의 light를 deterministic canonical sort하거나, scalar light encoder도 shared per-light encoder+aggregate 구조로 바꿔야 한다.

첫 implementation 비용을 줄이려면 현재 single-light 데이터에서는 K=1로 검증하고, 실제 two-light 데이터가 생길 때 shared-encoder sum을 활성화한다.

---

## 6. TokenLight 내부 fusion 위치

### 6.1 현재 실제 token shape

현재 480 RGB cached model의 대표 shape는 다음과 같다.

```text
source VAE latent       [B, 48, 1, 30, 30]
Wan patch size          (1, 2, 2)
source token grid       (1, 15, 15)
source tokens           [B, 225, 3072]
target tokens           [B, 225, 3072]
light scalar tokens     [B, 19, 3072]   # max_lights=2
baseline sequence       469 tokens
```

현재 source insertion 위치는 `model/tokenlight_wan.py:191-195`다.

### 6.2 기존 spatial-prefix는 smoke test로 사용 가능

`model/tokenlight_wan_spatial.py:60-186`에 이미 dense map용 `SpatialConditionEncoder`가 있고, `:385-401`에서 225개의 spatial prefix token을 append한다. 이 경로를 `moge` condition kind로 확장하면 전용 VAE 없이 빠른 smoke test가 가능하다.

다만 token append는 self-attention 비용을 크게 늘린다.

| 구조 | sequence length | baseline 대비 attention의 대략적 `L²` 비율 |
|---|---:|---:|
| source residual | 469 | 1.00× |
| spatial prefix 1개 | 694 | 2.19× |
| spatial prefix 2개 | 919 | 3.84× |

따라서 existing prefix는 연결 확인용이고 최종안은 source residual이다.

### 6.3 최종 residual fusion

Static encoder와 dynamic encoder는 분리한다.

\[
S'=S+g_sE_s(G_{scene})+g_lE_l(G_{light})
\]

```python
source_tokens, source_grid = _patch_to_tokens(
    dit,
    tokenlight_source_latents,
    batch,
)
source_tokens = _add_type_embedding(
    source_tokens,
    tokenlight_type_embedding,
    TOKENLIGHT_TYPE_SOURCE,
)

scene_delta = scene_geometry_encoder(
    scene_geometry,
    output_size=source_grid[1:],
).to(source_tokens.dtype)

light_delta = light_geometry_encoder(
    light_geometry,
    output_size=source_grid[1:],
).to(source_tokens.dtype)

scene_gate = torch.sigmoid(scene_gate_logit)
light_gate = torch.sigmoid(light_gate_logit)

source_tokens = (
    source_tokens
    + scene_gate * scene_delta
    + light_gate * light_delta
)
```

Geometry residual은 source와 동일한 15×15 위치에 있으므로 별도 RoPE를 추가하지 않는다. Source RoPE를 한 번만 적용하며 residual은 그 위치를 그대로 상속한다.

### 6.4 Encoder와 gate 초기화

기존 `SpatialConditionEncoder`의 CNN stem은 재사용할 수 있다. 권장 변경은 다음이다.

- Static/dynamic encoder를 별도 module로 둔다.
- Residual mode에서는 missing/drop output을 정확한 zero로 만든다.
- CNN feature를 15×15로 맞춘 뒤 3072 dimension으로 projection한다.
- Projection 뒤 RMSNorm 또는 LayerNorm을 적용할 수 있다.
- Branch마다 learnable scalar gate를 둔다.

안전한 초기화는 둘 중 하나다.

```text
A. projection small-random + sigmoid gate(-4) ≈ 0.018
B. final projection exact-zero + gate=1
```

Projection과 gate를 모두 exact-zero로 두면 초기에는 stem으로 gradient가 흐르지 않는다. 둘 중 하나만 zero에 둔다.

학습 로그에는 다음을 반드시 남긴다.

```text
RMS(scene_delta) / RMS(source_tokens)
RMS(light_delta) / RMS(source_tokens)
scene_gate, light_gate
각 encoder/gate gradient norm
```

---

## 7. CFG와 dropout

정확한 CFG 구조는 다음이다.

| condition | positive | null-light/negative |
|---|---|---|
| Source RGB | 유지 | 유지 |
| Static geometry `Gscene` | 유지 | 유지 |
| Dynamic map `Glight(L)` | 유지 | zero/null |
| Scalar Lightoken | real light | null light |

즉 static geometry는 light condition이 아니다. CFG negative에서 제거하면 안 된다.

Dynamic map은 반드시 scalar light token과 같은 `tokenlight_drop_light` mask를 공유한다.

```python
drop = tokenlight_drop_light.view(batch, 1, 1, 1)
light_geometry = torch.where(
    drop,
    torch.zeros_like(light_geometry),
    light_geometry,
)
```

Encoder bias 때문에 zero input에서도 nonzero residual이 나올 수 있다. Light 정보를 완전히 막으려면 encoder 출력에도 같은 mask를 곱한다.

```python
light_delta = light_delta * (~drop_light).view(batch, 1, 1)
```

현재 `train_tokenlight_spatial_safe.py:573-578`은 spatial condition 전체를 별도 확률로 drop하고 negative branch에서 항상 제거한다. Static+dynamic geometry를 하나의 spatial map으로 합쳐 이 동작을 그대로 쓰면 CFG가 틀어진다. 두 branch를 분리하고 다음을 지킨다.

- `tokenlight_light_dropout=0`
- 외부 `tokenlight_cfg_drop_prob`가 scalar light와 dynamic map을 함께 제어
- Static geometry dropout은 필요할 때만 별도의 작은 robustness regularizer로 사용하며 CFG 의미와 분리

---

## 8. MoGe-3 precompute와 cache

### 8.1 현재 공식 상태

MoGe-3는 2026-08-18 공식 repository에 v3 코드와 ViT-L/ViT-G model 항목이 공개됐다. [2026-08-20, commit `74fbce054...` 기준 공식 README](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/README.md#L203-L260)와 app은 Hugging Face 모델을 가리키지만, 같은 commit의 [`infer.py`](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/scripts/infer.py#L76-L84)는 `--pretrained`를 생략한 v3 실행에서 아직 오래된 UsageError를 낸다. 따라서 checkpoint를 명시하는 것이 안전하다.

```bash
moge infer \
  --version v3 \
  --pretrained Ruicheng/moge-3-vitl \
  --input SOURCE_OR_DIRECTORY \
  --output OUTPUT_DIRECTORY \
  --fov_x 39.6 \
  --refine_steps 3 \
  --maps
```

대규모 cache 전에는 한 scene으로 model download와 FlexGEMM/Triton 환경을 먼저 검증한다.

### 8.2 API 사용 시 invalid 값을 명시적으로 처리한다

MoGe `infer(apply_mask=True)`는 invalid point/depth를 `inf`로 만든다. Conv input에 `inf`가 들어가면 전체 activation이 망가질 수 있으므로 custom precompute에서는 다음 방식이 안전하다. `force_projection`, metric scale 적용, masking의 실제 순서는 [공식 v3 `infer()` 구현](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v3.py#L251-L335)에서 확인할 수 있다.

```python
output = model.infer(
    image,
    fov_x=39.6,
    force_projection=True,
    apply_mask=False,
    refine_steps=3,
)

points = output["points"].float()
normals = output["normal"].float()
valid = (
    output["mask"].bool()
    & torch.isfinite(points).all(dim=-1)
    & (points[..., 2] > 0)
)

points = torch.where(valid[..., None], points, 0)
normals = torch.where(valid[..., None], normals, 0)
```

아래 shape는 unbatched `[3,H,W]` 이미지와 현재 선택한 공개 `Ruicheng/moge-3-vitl` checkpoint를 전제로 한다. Batch input이면 앞에 `B` 차원이 붙는다. `normal`과 metric scale 적용은 해당 head를 가진 checkpoint에서만 보장되며, 공개 MoGe-3 ViT-L/G에는 두 head가 포함되어 있다. 공식 output contract는 [v3 docstring](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v3.py#L197-L232)과 [README example](https://github.com/microsoft/MoGe#-minimal-code-example)에 명시되어 있다.

```text
points       [H,W,3] metric camera-space point
depth        [H,W]   camera-space z-depth
normal       [H,W,3] OpenCV camera normal
mask         [H,W]   valid mask
intrinsics   [3,3]   normalized intrinsics
```

Internal metric scale은 이미 `points/depth`에 적용되며 별도 반환되지 않는다.

### 8.3 권장 cache schema

같은 source를 24–64개의 target light가 공유하므로 scene당 한 파일만 둔다.

```text
moge_geometry.npz
├─ points_cv       float32 [H,W,3]
├─ normal_cv       float16 [H,W,3]
├─ depth_cv_z      float32 [H,W]
├─ valid           uint8   [H,W]
├─ intrinsics      float32 [3,3]
├─ fov_x_degrees   float32 scalar
└─ alpha_moge      float32 scalar  # 사용한 custom calibration 값
```

초기 검증에서는 point/depth를 float32로 유지한다. MoGe-3도 [fine-depth voxelization을 fp32로 강제](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v3.py#L56-L90)한다. 저장량이 문제가 된 뒤 float16/quantization ablation을 한다. `normal.png`와 colorized depth PNG는 visualization용이며 condition source로 쓰지 않는다.

Manifest/sidecar에는 최소한 다음 provenance를 남긴다.

```json
{
  "moge_geometry": "scenes/.../moge_geometry.npz",
  "moge_version": "v3",
  "moge_checkpoint": "Ruicheng/moge-3-vitl",
  "moge_refine_steps": 3,
  "geometry_coord_system": "opencv_camera_xright_ydown_zforward",
  "source_resolution": [480, 480],
  "fov_x_degrees": 39.6,
  "canonical_camera_origin": [0.0, -3.5, 0.0],
  "scale_policy": "...",
  "feature_schema_version": "tokenlight_moge_v1"
}
```

---

## 9. FOV, resize, crop, augmentation

현재 source/target은 480×480이고 camera horizontal FOV는 39.6°다. 최종 480×480 source를 먼저 만든 뒤 그 이미지에 대해 MoGe를 실행하고 `fov_x=39.6`을 전달한다.

Aspect-ratio-preserving resize는 FOV를 바꾸지 않는다. Center crop은 crop width에 맞춰 FOV를 갱신해야 한다.

\[
f_x=\frac{W_0}{2\tan(\phi_0/2)},\qquad
\phi_c=2\tan^{-1}\left(\frac{W_c}{2f_x}\right)
\]

Off-center crop은 principal point가 바뀐다. MoGe API는 full intrinsic matrix가 아니라 `fov_x`를 받으며 [공식 focal recovery](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/utils/geometry_torch.py#L103-L153)는 centered principal point, undistorted image, isometric x/y를 가정하므로 피하는 편이 좋다. 가로세로 비율을 무시한 square stretch도 피한다.

현재 cached training에는 random crop/flip이 없으므로 V1은 정렬이 단순하다. 향후 horizontal flip을 추가하면 다음을 모두 함께 바꾼다.

```python
points = torch.flip(points, dims=[-1])
points[:, 0] *= -1

normals = torch.flip(normals, dims=[-1])
normals[:, 0] *= -1
normals = torch.nn.functional.normalize(normals, dim=1, eps=1e-6)

light_position[:, 0] *= -1
valid = torch.flip(valid, dims=[-1])
```

Source, target, point, normal, valid에 정확히 같은 spatial transform을 적용한다. Raw point boundary를 bilinear resize하면 서로 다른 표면의 XYZ가 섞이므로 다음 원칙을 쓴다.

- Raw point: nearest/depth-aware resize 또는 최종 해상도에서 MoGe 재추론
- Normal: resize 후 normalize
- Valid mask: nearest
- Learned geometry feature: bilinear/adaptive pooling 가능

---

## 10. Same-scene dual-light / velocity-delta 학습과 결합

Geometry branch는 앞서 만든 same-scene pair 학습과 잘 맞는다.

```text
                         ┌─ light i ─ Glight_i ─ DiT ─ target i
source ─ MoGe ─ Gscene ──┤
                         └─ light j ─ Glight_j ─ DiT ─ target j
```

Pair 내부에서는 다음이 같아야 한다.

- Source RGB/source latent
- MoGe cache와 `Gscene`
- Noise와 timestep
- Crop/flip
- CFG drop 상태

달라지는 것은 light attrs와 그 light로 계산한 `Glight`뿐이다. Static encoder는 pair당 한 번 계산해 feature를 재사용할 수 있다.

Velocity-delta loss는 기존 설계대로 유지한다.

\[
L_{\Delta}=\rho\left[
(\hat v_j-\hat v_i)-(v_j^*-v_i^*)
\right]
\]

Geometry 도입과 delta loss를 한 번에 켜면 어느 요소가 효과를 냈는지 알기 어렵다. 먼저 pointwise flow loss로 geometry ablation을 마친 뒤 마지막에 delta loss를 결합한다.

CFG로 light를 drop한 pair에서는 scalar light와 `Glight_i/Glight_j`를 함께 drop하고 delta loss를 제외한다.

---

## 11. 학습과 checkpoint 설계

### 11.1 무엇을 학습할지

```text
MoGe-3                         frozen
Wan VAE                        frozen
Scene geometry encoder         train
Light geometry encoder         train
Scene/light gates              train
Existing Lightoken encoder     load, low LR 또는 초기 freeze
Wan DiT LoRA                   load, low LR
```

권장 초기 LR은 다음 범위다.

```text
new geometry encoders/gates    1e-4
existing DiT LoRA              2e-5 ~ 5e-5
existing light/type encoder    2e-5 ~ 5e-5
```

현재 training runtime이 모든 trainable parameter에 같은 LR을 적용한다면 optimizer param group을 추가해야 한다. 첫 수백 step 동안 기존 light encoder/LoRA를 잠깐 freeze하고 geometry branch만 안정화하는 방법도 비교할 수 있다.

### 11.2 Warm-start

기존 RGB checkpoint에서 다음을 load한다.

- DiT LoRA
- `light_encoder`
- 기존 TokenLight type embedding

Residual fusion은 source token에 더하므로 새 token type은 필요 없다. 새 checkpoint에는 아래 module을 별도로 export/load해야 한다.

```text
tokenlight_scene_geometry_encoder.*
tokenlight_light_geometry_encoder.*
tokenlight_scene_geometry_gate.*
tokenlight_light_geometry_gate.*
```

기존 optimizer state까지 resume하기보다 RGB checkpoint를 weight warm-start로 읽고 새 optimizer를 만드는 편이 안전하다.

### 11.3 Baseline timestep semantics를 보존한다

현재 base one-frame TokenLight와 safe spatial/PBR 구현은 clean-prefix timestep 처리에 차이가 있다. Geometry 실험에서 이 동작까지 동시에 고치면 geometry 효과와 timestep bugfix 효과가 섞인다.

첫 residual ablation은 기존 RGB baseline의 timestep semantics를 그대로 복제한다. Clean-prefix 수정은 별도 control 실험으로 둔다.

---

## 12. 권장 실험 순서

| 실험 | 입력/구조 | 확인할 것 |
|---|---|---|
| A0 | 기존 RGB baseline을 같은 step만큼 fine-tune | 추가 step control |
| A1 | GT static 8ch, source residual | encoder/fusion 자체 효과 |
| A2 | A1 + light direction | 상대 위치 효과 |
| A3 | A2 + distance | near-field 효과 |
| A4 | A3 + signed `N·L` + attenuation | local shading/그림자 방향 |
| A5 | MoGe + GT-fitted scale | MoGe shape noise만 분리 |
| A6 | MoGe + deployable scale calibration | 실제 사용 성능 |
| A7 | A6 + low-frequency coordinate encoding | encoding 이득 |
| A8 | A6 + same-scene velocity-delta | multi-light consistency |

`A0` control이 중요하다. Geometry 모델만 baseline checkpoint에서 추가 fine-tune하고 원 baseline과 비교하면 extra step 효과를 geometry 효과로 오인할 수 있다.

평가는 다음을 함께 본다.

- Full/object/background PSNR, SSIM, LPIPS
- Renderer `inf_mask`, `shadow_mask`, boundary에서 RGB RMSE/MAE/PSNR
- Same-scene target pair의 delta error
- 연속 light trajectory의 first/second difference error
- Geometry branch gate와 residual RMS
- Scale error, point reprojection error, normal angular error

---

## 13. 필수 좌표/구현 테스트

### 13.1 축 basis test

```text
A [1,0,0] = [1,0,0]     OpenCV right   -> canonical +x
A [0,1,0] = [0,0,-1]    OpenCV down    -> canonical -z
A [0,0,1] = [0,1,0]     OpenCV forward -> canonical +y
```

### 13.2 Camera/light 변환 test

```text
canonical camera O=[0,-3.5,0]
canonical light L=[x,y,z]
camera-frame light=[x,-z,y+3.5]
```

Center pixel ray가 camera frame `+Z`, canonical frame `+y`인지 확인한다.

### 13.3 Renderer metadata round-trip

각 scene에서 다음을 확인한다.

\[
L_w\simeq C_t+sR(L_c-C)
\]

\[
O_w\simeq C_t+sR(O_c-C)
\]

### 13.4 Frame invariance

동일한 alignment mode에서 camera-frame과 canonical-frame으로 계산한 결과가 같아야 한다. Free translation mode라면 §3의 `A^T(t-O_can)` 보정까지 포함한다.

```text
distance_cv ≈ distance_can
ndotl_cv ≈ ndotl_can
A @ direction_cv ≈ direction_can
```

### 13.5 Normal polarity

Official output은 normal이 OpenCV camera coordinate라고 명시하지만 front/back polarity는 실제 cache 한 장에서 검증한다.

```python
view_dir = torch.nn.functional.normalize(-points_cv, dim=-1)
score = (normals_cv * view_dir).sum(-1)[valid].median()
```

Visible front face에서 지속적으로 음수면 convention을 한 번 뒤집는다. `abs(N·L)`로 숨기면 물리적 방향 정보가 사라지므로 사용하지 않는다.

### 13.6 CFG leakage test

같은 source/static geometry에서 null-light branch의 scalar light와 dynamic map을 바꿔도 output condition tensor가 완전히 동일해야 한다. Static map은 positive/negative에서 bitwise 동일해야 한다.

### 13.7 Pair consistency test

같은 scene의 서로 다른 light row는 다음이 같아야 한다.

```text
moge cache path
static geometry tensor
source latent
coordinate schema/scale policy
```

Dynamic map만 light에 따라 달라져야 한다.

### 13.8 Visual debug sheet

각 scene에서 다음을 한 장에 저장한다.

```text
source | point X/Y/Z | normal | valid | log-depth
light direction X/Y/Z | distance | signed N·L | attenuation
GT shadow/direct masks
```

그림을 보기 전에 축·원점·scale histogram도 함께 기록한다.

---

## 14. 파일 구성 제안

기존 baseline 파일을 보존하고 geometry용 경로를 별도로 만드는 것이 안전하다.

```text
scripts/precompute_moge3_geometry.py
model/tokenlight_geometry_condition.py
model/tokenlight_wan_geometry.py
model/train_tokenlight_geometry_safe.py
scripts/infer_manifest_geometry.py
configs/train_480/rgb_moge_geometry_*.json
```

역할은 다음처럼 나눈다.

- `precompute_moge3_geometry.py`: source별 MoGe inference, FOV, cache, provenance
- `tokenlight_geometry_condition.py`: 축/scale 변환, feature builder, encoder, gates
- `tokenlight_wan_geometry.py`: source-residual fusion이 추가된 별도 model function
- `train_tokenlight_geometry_safe.py`: 두 encoder optimizer/checkpoint/CFG wiring
- `infer_manifest_geometry.py`: 동일 coordinate schema로 cache load 및 dynamic map 생성

기존 `SpatialConditionEncoder`의 CNN 구현과 checkpoint export 패턴은 재사용하되, 기존 RGB trainer와 model function은 직접 수정하지 않는 편이 ablation과 rollback에 유리하다.

---

## 15. 최종 권장 V1

지금 바로 하나를 선택한다면 다음 구성이 가장 방어적이다.

```text
Geometry source       MoGe-3 ViT-L, source별 offline cache
FOV                   metadata의 39.6°, 최종 480 image에 실행
Dense frame           MoGe native OpenCV camera frame
Scale                 GT oracle로 먼저 검증, 이후 fixed-rig scale head
Static encoding       calibrated XYZ/3.5 + N + logZ + valid = 8ch
Dynamic encoding      direction + log-distance + signed N·L
                      + bounded attenuation + valid = 7ch/light
Fourier               dense branch에는 없음
Global Lightoken      기존 canonical scalar Fourier token 유지
Fusion                source token residual, two separate encoders/gates
Gate init             sigmoid(-4) ≈ 0.018
CFG                   static 유지, dynamic과 scalar light 동시 drop
Training              pointwise FM으로 geometry ablation 후 delta loss 결합
```

이 설계의 핵심은 MoGe를 넣는다는 사실 자체가 아니다. **MoGe point와 light가 같은 origin, axis, scale을 사용하도록 만든 뒤, 동일 source geometry를 모든 light target이 공유하게 하는 것**이다. 이것이 조명이 바뀔 때 모델이 scene shape를 매번 다르게 해석하는 문제를 직접 겨냥한다.

MoGe point map은 image pixel마다 visible surface를 나타내는 sparse shell이며([MoGe-3 §3.2.1](https://arxiv.org/html/2607.17967#S3.SS2.SSS1)), 보이지 않는 뒷면, 화면 밖 occluder, watertight volume을 제공하지 않는다. 논문도 boundary ambiguity와 잔여 fly-point를 [한계로 명시한다](https://arxiv.org/html/2607.17967#S5). 따라서 정확한 cast-shadow visibility를 보장하지는 않는다. 이 구조의 현실적인 목표는 geometry consistency와 local light-surface relation을 강화하는 것이며, hard visibility가 추가로 필요하다는 증거가 나온 뒤 sparse point occlusion/visibility module을 V2로 검토한다.

---

## 16. 공식 자료와 로컬 근거

### MoGe 공식 자료

- [MoGe 공식 repository와 MoGe-3 pretrained model 목록](https://github.com/microsoft/MoGe)
- [MoGe-3 논문](https://arxiv.org/abs/2607.17967)
- [MoGe-3 공식 v3 구현](https://github.com/microsoft/MoGe/blob/main/moge/model/v3.py)
- [MoGe-3 ViT-L checkpoint](https://huggingface.co/Ruicheng/moge-3-vitl)
- [MoGe-3 ViT-G checkpoint](https://huggingface.co/Ruicheng/moge-3-vitg)
- [MoGe 공식 inference CLI](https://github.com/microsoft/MoGe/blob/main/moge/scripts/infer.py)
- [MoGe-2 논문](https://arxiv.org/abs/2507.02546)

### 현재 workspace 근거

- `model/physical_tasks/data.py:105-136`: right-forward-up ray, point/light relation 재구성
- `scripts/build_final_objaverse_light_mask_infer_manifest.py:151-166`: canonical light position을 attrs로 변환
- `model/lightoken_encoder.py:107-193`: 기존 scalar Gaussian Fourier encoding
- `model/tokenlight_wan.py:183-216`: target/source/light token 구성
- `model/tokenlight_wan_spatial.py:60-186`: 기존 dense spatial CNN encoder
- `model/tokenlight_wan_spatial.py:385-401`: 기존 spatial-prefix append 방식
- `model/train_tokenlight_spatial_safe.py:554-578`: spatial input/dropout wiring
- `data/unseen_fixed32_random2_png/scenes/scene_002502/meta.json`: canonical camera, similarity transform, canonical lights
