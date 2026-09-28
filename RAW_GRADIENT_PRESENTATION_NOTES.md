# Raw-gradient 실험 발표 정리

## 지표 계산 방법

### Probe와 Monte Carlo 반복

- Task마다 seed로 고정한 20개 demonstration episode를 사용했다.
- Episode마다 full 16-step action chunk를 만들 수 있는 구간의 20%, 40%, 60%, 80% 지점에서 probe를 만들었다. 따라서 Task당 80 probes다.
- 한 probe는 현재 observation/state/language와 16-step expert action chunk로 구성된다.
- 같은 Task의 probe와 전처리는 모든 checkpoint에서 동일하다.
- Flow noise와 flow time은 4회 독립 반복했다. 같은 `(task, probe, repeat)`의 noise/time은 checkpoint 사이에서 동일하게 사용해 checkpoint 비교를 paired comparison으로 만들었다. Task가 다르면 noise/time도 독립적이다.
- 각 반복에서 `epsilon ~ N(0,I)`, `beta ~ Beta(1.5,1.0)`, `t=(0.999-beta)/0.999`를 사용했다.

### Action과 flow-matching loss

LIBERO의 position delta, rotation delta, gripper action은 dataset min/max 통계로 각각 다음과 같이 변환된다.

`a_norm = 2 * (a - min) / (max - min) - 1`.

`min == max`인 차원은 0으로 둔다. 이 정규화는 서로 단위와 범위가 다른 action 차원이 MSE와 gradient를 한쪽에서 지배하지 않게 하고, 원래 checkpoint 학습 파이프라인과 같은 입력 공간을 유지하기 위해 필요하다.

정규화된 expert action을 `a`, Gaussian noise를 `epsilon`이라 하면:

`x_t=(1-t)epsilon+t*a`, `v_target=a-epsilon`.

Action head는 `v_theta(x_t,t,condition)`를 예측한다. Probe loss는:

`L = sum M * (v_theta-v_target)^2 / sum M`.

`M`은 action mask다. Horizon 16, 최대 action dimension 32 중 LIBERO의 실제 action 차원만 loss에 포함하며 padding은 제외한다. Mask 합으로 나누므로 valid action element당 평균 MSE다.

### Raw gradient

반복 `r`, Task `j`, checkpoint `Ck`에서:

`g_j^(r)(Ck) = grad_theta_AH [ (1/80) sum_i L_i ]`.

`theta_AH`는 active LIBERO action-head parameter다. Frozen VLM backbone에는 gradient를 계산하지 않는다. BF16 forward/backward kernel을 사용하되 action-head parameter와 gradient 누적은 FP32, dot/norm 합산은 FP64다. Gradient clipping, AdamW moment, adaptive scaling, learning rate, weight decay는 적용하지 않은 raw loss gradient다.

네 noise/time 반복의 평균 gradient를:

`gbar_j(Ck) = (1/4) sum_r g_j^(r)(Ck)`

로 정의한다. 주 cosine은 반복별 cosine의 평균이 아니라 `gbar` 사이의 cosine이다. 이는 stochastic flow noise/time에 대한 expected-loss gradient를 근사하기 위해서다.

### Cross-task conflict cosine

Task `k` 학습 직전 checkpoint에서:

`c_conflict(j,k) = dot(gbar_j(C(k-1)), gbar_k(C(k-1))) / (norm(gbar_j) norm(gbar_k))`, `j<k`.

L2 norm으로 나누는 이유는 gradient 크기를 제거하고 방향만 비교하기 위해서다. 정규화하지 않은 dot product는 방향과 크기가 섞인다. 음수는 vanilla gradient-descent의 국소적 raw-gradient conflict를 뜻한다. 전체 action head를 하나의 벡터로 취급하며 parameter/layer별 추가 정규화는 하지 않았다.

### Same-task task-end cosine

`c_drift(j,k) = cos(gbar_j(Cj), gbar_j(Ck))`, `k>=j`.

`Cj`는 Task `j`를 막 학습한 checkpoint다. Cosine의 L2 정규화 목적은 위와 같다. 이 값은 같은 Task의 누적 gradient rotation이며 cross-task conflict가 아니다.

### Mean raw-gradient norm

`n(j,k)=norm(gbar_j(Ck))_2`.

Gradient norm 자체는 정규화하지 않았다. 모든 cell이 같은 whole action-head parameter 집합과 같은 masked objective를 사용하므로 직접 비교했다. Heatmap 표시에서만 `log10(n(j,k))`를 사용했다. 값의 범위가 약 0.1부터 수 단위까지 넓기 때문에 작은 task-end norm과 큰 post-training norm을 한 그림에 보이기 위한 시각화 변환이다. 17.78--43.49배 비율은 log 값이 아니라 원래 linear norm의 비율로 계산했다.

### Old-task flow-loss change

`Delta L(j,k) = mean_r L_j^(r)(Ck) - mean_r L_j^(r)(C(k-1))`.

별도의 ratio 또는 표준화는 하지 않았다. 같은 probe, preprocessing, noise/time과 같은 normalized-action loss를 양 endpoint에서 사용하므로 직접 차이를 취했다. 학습 직후 loss가 매우 작아 ratio가 불안정할 수 있으므로 absolute difference를 사용했다.

### 보조 신뢰도 지표

`SE_MC = sqrt( sum_r norm(g_j^(r)-gbar_j)^2 / (R(R-1)) )`, `R=4`.

표시한 신호비는 `norm(gbar_j)/SE_MC`다. `norm(gbar_j) < 2*SE_MC`를 low-signal 휴리스틱으로 표시했다. 이는 noise/time Monte Carlo 안정성 진단이며 통계적 유의성 검정이나 여러 training seed의 신뢰구간이 아니다.

## 연구 질문

1. 새 Task와 이전 Task의 raw gradient가 학습 직전에 충돌하는가?
2. 후속 Task 학습 후 이전 Task의 gradient 방향과 크기는 어떻게 변하는가?
3. 이러한 기하 변화가 flow loss 및 rollout success 변화와 함께 나타나는가?

## 그림 1: cross-task conflict와 망각 결과

모든 행은 전이 `C(k-1) -> Ck`, 모든 열은 이전 Task `j < k`를 나타낸다.

### Before training: old vs incoming gradient

`cos(g_j(C(k-1)), g_k(C(k-1)))`이다. 같은 파라미터 위치에서 이전 Task와 incoming Task를 비교한다. 음수는 raw-gradient conflict, 양수는 정렬이다. 전체 action head에서 안정적인 음수는 Task 1--3 한 pair뿐이며 cosine은 -0.1936이다. 나머지 14/15 pair는 네 noise/time 반복에서 모두 양수다.

### Same old task: before vs after gradient

`cos(g_j(C(k-1)), g_j(Ck))`이다. 같은 이전 Task의 gradient가 새 Task 학습 전후 얼마나 회전했는지 측정한다. 이것은 conflict가 아니라 adjacent gradient drift다. 1은 방향 유지, 0은 거의 직교, 음수는 방향 반전이다.

### Old-task flow loss: after - before

`L_j(Ck) - L_j(C(k-1))`이다. 양수는 이전 Task의 masked flow-matching loss 악화, 음수는 감소다. 15개 전이 중 10개에서 증가했다.

### Old-task success drop

`100 * (S_j(C(k-1)) - S_j(Ck))` percentage points다. 양수는 rollout 성능 하락이다. 0은 보존을 보장하지 않는다. 이미 success가 0이면 추가 하락이 보이지 않는 floor effect가 있다.

## 그림 2: same-task raw-gradient geometry

가로축은 평가 Task `j`, 세로축은 checkpoint `Ck`다. Task가 학습된 이후의 `j <= k`만 평가했으므로 lower triangle만 값이 있다.

### Same-task cosine vs task-end checkpoint

`cos(g_j(Cj), g_j(Ck))`이다. Task를 막 학습한 시점을 기준으로 한 누적 gradient rotation이다. 대부분 후속 checkpoint에서 0 근처이므로, 이전 Task를 다시 개선하는 방향이 task-end 시점과 크게 달라졌다. 이는 cross-task conflict의 증거가 아니다.

### Same-task cosine vs previous checkpoint

`cos(g_j(C(k-1)), g_j(Ck))`이다. 한 Task 전이 동안 발생한 local gradient rotation이다. 그림 1의 오른쪽 위 패널과 같은 지표다.

### log10 norm of mean raw gradient

`log10(||mean_r g_j^(r)(Ck)||_2)`다. 예를 들어 -0.66은 norm 약 0.22, 0.59는 약 3.9다. 바로 다음 Task 학습 후 이전 Task gradient norm은 task-end 대비 17.78--43.49배 증가했다. task-end에서 작던 잔여 gradient가 후속 학습 후 크게 증가했음을 뜻한다.

### Mean-gradient norm / MC vector SE

`||mean gradient|| / MC vector standard error`다. 네 noise/time 반복에서 평균 gradient 신호가 Monte Carlo 변동보다 얼마나 큰지 보는 진단 지표다. 2 미만을 low-signal 휴리스틱으로 사용했다. 이것은 통계적 유의성 검정이나 학습 seed 신뢰구간이 아니다. Task 5/checkpoint 5가 1.98로 경계에 있지만 후속 conflict pair에는 사용되지 않는다.

## 발표용 핵심 관찰

- 안정적인 음의 raw-gradient conflict는 1/15 pair뿐이다.
- 두 패널을 같은 cell끼리 결합하면 `negative conflict + loss increase` 1개, `positive alignment + loss increase` 9개, `positive alignment + non-positive loss change` 5개, `negative conflict + non-positive loss change` 0개다.
- 즉 loss가 증가한 10개 전이 중 9개는 학습 직전 raw gradient가 양의 정렬이었다. 초기 raw-gradient conflict만으로 Task 전체 학습 후 loss 악화를 설명할 수 없다.
- Task 0--1은 conflict cosine +0.2982인데 loss가 +0.6195 증가하고 success가 100 percentage points 하락했다.
- Task 2--3은 conflict cosine +0.2178인데 loss가 +0.5033 증가하고 success가 93.3 percentage points 하락했다.
- 유일한 conflict인 Task 1--3은 cosine -0.1936이고 loss는 +0.4989 증가했지만, Task 1 success가 이미 0이어서 추가 행동 하락을 평가할 수 없다.
- 따라서 이 pilot에서는 음의 raw-gradient conflict가 관찰된 망각의 필요조건이 아니다. 유일한 음수 pair는 success가 이미 0인 floor 상태이므로 행동적 충분조건 여부는 판단할 수 없다.
- 후속 학습 후 task-end 기준 gradient cosine은 대부분 0 근처이고 norm은 크게 증가한다. 망각과 함께 이전 Task의 복구 방향과 요구되는 보정량이 크게 변했다.

## Task 학습 전후 fixed-probe raw conflict

- AdamW 변환을 적용하지 않은 fixed-probe 평균 raw gradient를 같은 checkpoint에서 비교했다.
- Task 학습 전 `C(k-1)`에는 음수 cosine이 1/15 pair였다: old Task 1 vs incoming Task 3, -0.1936.
- Task 학습 후 `Ck`에도 음수 cosine은 1/15 pair였지만 old Task 1 vs learned Task 2의 -0.0157로 거의 0이었다.
- Task 5 학습 후 pair는 `g_5(C5)`의 MC signal ratio가 1.98이라 low-signal로 표시했다.
- 이 값은 checkpoint에서 재계산한 offline raw gradient이며 당시 AdamW에 입력된 historical minibatch gradient가 아니다.
- 결과: `outputs/analysis/libero_spatial/none/raw_conflict_task0_to_5_pilot_20260927/phase_start_end_raw_conflict.csv` 및 `figures/04_phase_start_end_raw_conflict.png`.

## 권장 슬라이드 구성

### 슬라이드 1: 측정 정의

- Cross-task conflict: `cos(g_old(theta), g_new(theta))`, 같은 checkpoint.
- Same-task drift: `cos(g_old(C_before), g_old(C_after))`, 다른 checkpoint.
- 두 값을 혼용하지 않는다는 점을 식과 작은 도식으로 제시한다.

### 슬라이드 2: conflict가 망각을 설명하는가?

- 그림 1의 왼쪽 위, 왼쪽 아래, 오른쪽 아래 패널을 사용한다.
- headline: `Only 1/15 pairs shows stable negative raw-gradient conflict.`
- Task 0--1과 Task 2--3을 표시해 양의 alignment에서도 큰 loss/success 악화가 발생했음을 강조한다.

### 슬라이드 3: 망각 후 old-task gradient geometry

- 그림 2의 왼쪽 위와 왼쪽 아래를 사용한다.
- headline: `Old-task recovery gradients rotate and grow after sequential training.`
- task-end baseline cosine이 대부분 0 근처이고 바로 다음 Task 이후 norm이 17.78--43.49배 증가한 점을 제시한다.
- SNR 패널은 본문보다 appendix 또는 reliability slide에 둔다.

### 슬라이드 4: 결론과 한계

- 결론: raw conflict는 일부 존재하지만 전체 망각을 설명하지 않는다.
- 범위: frozen VLM, trainable action head, 80-probe pilot, noise/time 반복 4회.
- raw gradient는 AdamW update가 아니므로 optimizer-level 인과를 주장하지 않는다.
- rollout success는 기존 30-episode 평가이며, success 0에는 floor effect가 있다.
- 모든 valid frame과 episode-cluster uncertainty를 사용한 최종 검증이 필요하다.
