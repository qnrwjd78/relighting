# Shared-Scene Dual-Light Flow 모델 설계

정확히는 모델을 **공유 scene encoder + 두 개의 light query를 처리하는 Siamese DiT** 구조로 학습하는 방식입니다.

## 모델 구조

```text
                         ┌─ Light Encoder(c_i) ─ L_i ─┐
Source image ─ VAE ─ S ─┤                            ├─ Shared DiT ─ v̂_i
                         │  noisy target latent z_i ──┘
                         │
                         ├─ Light Encoder(c_j) ─ L_j ─┐
                         │                            ├─ Shared DiT ─ v̂_j
                         └── noisy target latent z_j ─┘
                                      │
                              v̂_j - v̂_i
                                      │
                             Target delta와 비교
```

핵심은 다음과 같습니다.

- Source scene latent $S$는 하나입니다.
- 조명 $i,j$는 같은 Lightoken Encoder를 통과합니다.
- DiT도 하나이며 두 branch가 모든 weight를 공유합니다.
- 두 branch 사이에 cross-attention을 넣지는 않습니다.
- 두 결과의 차이를 loss로 연결합니다.
- inference에서는 기존처럼 source와 조명 하나만 넣습니다.

즉, 모델 두 개를 만드는 것이 아니라 학습할 때만 동일 모델을 두 조명 조건으로 동시에 호출하는 Siamese 구조입니다. 실제 구현은 두 샘플을 batch dimension에 붙여 DiT를 한 번 호출하면 됩니다.

## 모델 안에서 실제로 일어나는 계산

같은 scene의 source에서:

$$
S=E_{\text{source}}(I_{\text{source}})
$$

두 조명을 각각 embedding합니다.

$$
L_i=E_{\text{light}}(x_i,y_i,z_i,\lambda_i,\ldots)
$$

$$
L_j=E_{\text{light}}(x_j,y_j,z_j,\lambda_j,\ldots)
$$

같은 noise와 timestep으로 두 target latent를 noisy latent로 만듭니다.

$$
z_i=(1-\sigma)x_i+\sigma\epsilon
$$

$$
z_j=(1-\sigma)x_j+\sigma\epsilon
$$

그리고 같은 DiT가 각각 velocity를 예측합니다.

$$
\hat v_i=F_\theta(z_i,S,L_i,\sigma)
$$

$$
\hat v_j=F_\theta(z_j,S,L_j,\sigma)
$$

여기서 $F_\theta$가 현재 우리 TokenLight DiT입니다. 새로운 renderer network를 하나 더 만드는 게 아닙니다.

## 학습 loss

모델은 두 가지를 동시에 학습합니다.

### 1. 각 조명의 absolute relighting

$$
L_{\text{abs}}
=
\operatorname{MSE}(\hat v_i,v_i^*)
+
\operatorname{MSE}(\hat v_j,v_j^*)
$$

기존 학습과 같습니다.

### 2. 조명이 이동했을 때 appearance 변화

$$
L_\Delta=
\operatorname{MSE}
\left(
(\hat v_j-\hat v_i)
-
(v_j^*-v_i^*)
\right)
$$

최종 loss는:

$$
L=L_{\text{abs}}+\lambda_\Delta L_\Delta
$$

입니다.

우리 scheduler는 $v^*=\epsilon-x$ convention이라, shared noise를 사용하면:

$$
v_j^*-v_i^*=x_i-x_j
$$

가 됩니다. 따라서 실제 구현에서는 부호를 직접 계산하지 않고 `target_v_j - target_v_i`를 그대로 사용해야 합니다.

## 왜 두 branch가 서로 정보를 주고받지 않게 하나

두 target을 동시에 attention에 넣으면 모델이 다른 target 이미지를 참고해 결과를 맞추는 식으로 학습할 수 있습니다. 그러면 inference에서 조명 하나만 넣었을 때 학습 구조와 달라집니다.

그래서:

- 두 branch는 source만 공유
- 서로 다른 light condition 사용
- DiT weight만 공유
- 관계는 loss에서만 연결

하는 구조가 안전합니다.

이렇게 하면 inference 구조가 변하지 않습니다.

```text
Source + 원하는 light position + noise → 기존 DiT → relighted image
```

## 우리 모델에서 변경되는 부분

모델 모듈 기준으로는 다음과 같습니다.

| 모듈 | 변경 |
|---|---|
| Frozen VAE | 변경 없음 |
| Source token encoder | 변경 없음 |
| Lightoken Encoder | 변경 없음 |
| Wan DiT + LoRA | 변경 없음 |
| Decoder/inference | 변경 없음 |
| Training forward | 두 조명을 묶어 호출 |
| Training loss | velocity-delta loss 추가 |

즉 checkpoint parameter 구조도 그대로입니다. Pair loss가 Lightoken Encoder와 DiT LoRA까지 gradient를 전달하면서, “조명 위치가 바뀌면 출력이 어떤 방향으로 변해야 하는가”를 학습시킵니다.

## Batch 8일 때 권장 형태

첫 모델은 다음처럼 구성하는 것이 좋습니다.

```text
Pair A: scene_A, light_i / scene_A, light_j
Pair B: scene_B, light_i / scene_B, light_j
Singleton: scene_C, scene_D, scene_E, scene_F
```

- absolute loss: 8개 모두
- delta loss: 두 pair만
- 한 번의 DiT forward
- 총 batch size는 계속 8
- pair fraction은 50%

처음부터 4 pair로 전부 채우면 batch당 고유 scene이 8개에서 4개로 줄어듭니다. 그래서 2 pair+4 singleton으로 먼저 시작하고, 이후 full-pair를 비교하는 것을 추천합니다.

## 정말 새로운 모듈을 추가하고 싶다면

후속 버전에서는 `Scene-Conditioned Delta Transport Adapter`를 추가할 수 있습니다.

$$
\hat d_{ij}
=
T_\phi(S,L_i,L_j,L_j-L_i,\sigma)
$$

이 adapter가 직접 “조명 $i\rightarrow j$의 latent 변화 map”을 예측합니다.

$$
L_{\text{transport}}
=
\operatorname{MSE}
\left(
\hat d_{ij},
v_j^*-v_i^*
\right)
$$

$$
L_{\text{consistency}}
=
\operatorname{MSE}
\left(
\hat v_j-\hat v_i,
\hat d_{ij}
\right)
$$

하지만 첫 모델부터 이 head를 넣는 것은 추천하지 않습니다. Delta head만 정답을 잘 맞추고 실제 DiT 출력은 좋아지지 않을 수 있기 때문입니다. 먼저 shared-DiT의 output velocity 차이에 직접 loss를 거는 것이 더 강하고 단순합니다.

## 결론

제가 제안하는 실제 모델은 다음과 같습니다.

> 현재 TokenLight 모델을 그대로 두고, 학습할 때 같은 source를 공유하는 두 light-conditioned DiT branch를 만든 뒤, 두 velocity 출력 차이를 GT velocity 차이와 맞추는 shared-weight Siamese flow 모델

모델 구조를 새로 만드는 것보다는 학습 그래프를 두 갈래로 확장하는 설계입니다.
