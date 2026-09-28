# Task 0 gradient 변화와 Task 1 AdamW 간섭 실험

작성일: 2026-09-27. 상태: Task 0→1 및 Task 0~5 raw gradient 비교 실험 완료. 단일 step audit 재학습은 보류했다.

## 1. 연구 문제

Task 1 학습 이후 Task 0 gradient 방향이 얼마나 달라지는가? Task 1 학습 중 실제 optimizer update가 Task 0 flow-matching loss를 증가시키는가? 이 현상이 Task 0 rollout success 저하와 함께 나타나는가?

Task 0 학습 완료 파라미터를 θ₀, Task 1 학습 완료 파라미터를 θ₁로 정의한다. 학습 중 단일 step의 파라미터는 별도로 θₜ, θₜ₊₁로 표기한다. g₀(θ)는 고정된 Task 0 probe loss의 gradient다.

## 2. 관찰: gradient 방향 변화

주 지표는 cos(g₀(θ₀), g₀(θ₁))다. 같은 목적함수에서 파라미터 위치에 따른 gradient 방향 변화를 측정한다. 이 값만으로 loss landscape 자체가 바뀌었다거나 Task 1의 gradient conflict가 망각을 유발했다고 결론 내리지 않는다.

함께 기록할 항목:

- 두 gradient norm, Task 0 flow-matching loss 및 그 차이.
- 두 checkpoint의 Task 0 rollout success와 평가 episode 수. 가능한 경우 동일 초기 상태 및 평가 seed로 비교한다.
- noise/time 반복별 cosine, norm, loss와 반복 평균 gradient 사이의 cosine.
- 같은 checkpoint에서 noise/time 반복 간 gradient 변동 및 mean-gradient norm 대비 Monte Carlo 오차.

같은 demo, frame, action mask, 전처리, loss reduction, 평가 모드, 파라미터 범위를 사용한다. 각 반복에서는 두 checkpoint에 동일한 flow-matching noise와 time을 사용한다. 반복 사이에는 seed를 변경한다. 동일 seed 재실행은 재현성 점검이며, noise 불확실성 측정과 구분한다.

작은 norm 자체보다 gradient 신호 대비 noise 크기를 평가한다. norm이 0이면 cosine은 정의되지 않는다. 신호가 약하면 cosine 해석을 보류한다. noise 반복은 독립 학습 seed 실험을 대체하지 않는다.

## 3. 원인 가설과 검증 방법

가설: Task 1의 실제 AdamW update가 Task 0 loss를 증가시키는 국소 방향을 포함한다.

선택한 Task 1 optimizer step마다 실제 파라미터 차이 Δθₜ = θₜ₊₁ − θₜ를 기록하고, 반드시 동일한 pre-step 파라미터에서 g₀(θₜ)를 계산한다.

측정값:

- pₜ = g₀(θₜ)ᵀΔθₜ: Task 0 loss 변화의 1차 예측.
- dₜ = L₀(θₜ₊₁) − L₀(θₜ): 동일 probe/noise/time으로 측정한 실제 loss 차이.
- rₜ = dₜ − pₜ: 1차 근사 잔차. 곡률과 수치 오차 등이 포함될 수 있다.
- gradient norm, update norm, cos(g₀(θₜ), Δθₜ), 반복별 부호와 변동.

pₜ > 0은 1차적으로 loss 증가 방향, pₜ < 0은 감소 방향이다. pₜ ≈ 0은 1차 효과가 작다는 뜻이며 실제 loss 변화가 없다는 뜻은 아니다. dₜ와의 일치 여부를 확인한다.

raw Task 1 gradient와의 cosine은 보조적인 기하 지표로만 사용한다. 실제 update에는 optimizer moment, scaling, clipping, learning rate, weight decay 및 구현의 수치 효과가 반영된다. 정확한 optimizer 종류와 설정도 실행 metadata에 남긴다.

## 4. 방법의 병목과 기존 코드

| 구성 요소 | 현재 코드에서 확인한 역할 | 이 실험에서의 처리 |
| --- | --- | --- |
| `scripts/run_gradient_first.py` | 고정 probe/noise/time, gradient drift, norm, loss, noise 신호 분석 | 1단계 재사용 후보. 현재 Task 0~5를 전제로 하므로 Task 0→1 전용 실행 범위 검토 필요 |
| `scripts/analyze_update_alignment.py` | task 종료 checkpoint 사이 누적 displacement에 시작점 gradient 투영 | 단일 optimizer step 검증과 분리하여 보조 분석으로 표시 |
| `gr00t/utils/optimizer_audit.py` | 선택 step의 pre-step action head와 실제 update 저장 | 단일 step 분석 입력으로 활용하되 저장 정밀도와 파라미터 범위 검증 필요 |
| `scripts/gr00t_sequential_finetune.py` | `--optimizer-audit`와 기록 step 설정 제공 | 기존 학습 설정을 유지하며 audit 수집에 사용 가능 |

주의할 구현 조건:

- 전체 task displacement에 대한 g₀(θ₀)ᵀ(θ₁−θ₀)는 각 step에서 계산한 g₀(θₜ)ᵀΔθₜ의 합과 일반적으로 다르다. 전체 task 구간에 대한 1차 근사는 잔차가 클 수 있다.
- audit 기본 step은 1000, 5000, 7500이다. 초기·중간·후기 측정 위치는 실행 전에 확정한다. 저장 비용이 크므로 모든 step 전체 벡터 저장을 기본값으로 삼지 않는다.
- 현재 audit은 pre-step 값을 BF16으로 저장한다. 원래 학습 파라미터가 FP32라면 정확한 pre-step 복원을 보장하지 않는다. 원래 dtype 보존 또는 오차 검증이 필요하다.
- audit은 trainable action head와 gradient로 판별한 active category row를 저장한다. probe gradient와 update의 이름·shape·row 정렬을 확인하고, 누락된 파라미터가 Task 0 loss에 영향을 주지 않는지 검증한다. backbone 재사용은 불변성이 확인된 경우에만 허용한다.
- Task 0 probe 계산이 학습 gradient buffer, RNG, 모델 모드 또는 optimizer state를 바꾸지 않도록 한다. 저장된 snapshot에서 별도 분석하면 간섭을 줄일 수 있다.
- 최종 checkpoint만으로 과거 단일 step update를 복원할 수 없다. audit 파일 또는 인접 step 파라미터가 있는지 확인해야 한다.

## 5. 실험 증거와 해석 기준

1. 관찰 증거: gradient 회전이 noise 변동보다 안정적인지 확인하고 loss 및 success 변화와 함께 보고한다.
2. 국소 간섭 증거: pₜ가 양수이고 dₜ도 반복적으로 양수인지 확인한다. 부호가 불일치하면 근사 잔차와 수치 오차를 조사한다.
3. 범위 제한: 일부 step의 양의 투영은 해당 step의 국소 loss 간섭 증거다. 전체 학습의 순효과나 rollout 망각의 단독 원인까지 입증하지는 않는다.
4. 강한 인과 주장이 필요하면 후속 대조 실험을 설계한다. 예를 들어 간섭을 줄이는 개입과 대조군에서 Task 0 보존 및 Task 1 학습 성능을 함께 비교한다. 이 개입은 현재의 두 단계 관찰·검증과 별도 범위다.

먼저 Task 0→1을 대상으로 측정 및 재현성을 검증한 뒤, 필요에 따라 다른 task 전이와 독립 학습 seed로 확장한다. 구현 및 실행 범위는 사용자 확인 후 확정한다.

## 실행 기록 (2026-09-27)

- 기존 Task 0~9 checkpoint와 RTX 4090 사용 가능. 과거 optimizer audit 파일은 없다.
- 1단계: `scripts/run_gradient_first.py --num-tasks 2 --repeats 4 --batch-size 4 --output outputs/analysis/libero_spatial/none/task0_to_1_adamw_20260927` 실행 시작. demo 20개 × frame 4개, noise/time 반복 4회.
- `run_gradient_first.py`에 task 수 설정을 추가했으며 기존 기본값 6은 유지했다.
- audit pre-step 저장을 원래 parameter dtype 보존으로 수정했다. FP32 학습 파라미터를 BF16으로 낮춰 저장하는 오차를 방지한다.
- 기존 trainer_state 실측 시간: Task 0 4635.2474초, Task 1 4708.1149초, Task 2 4665.1769초, Task 3 4631.6682초. task당 10,000 step, batch 16, AdamW 8-bit, GPU 4090 기준.
- Task 0 checkpoint 재사용 후 Task 1~3 재학습의 합은 14004.96초(약 3시간 53분). Task 0부터 모두 학습하면 18640.21초(약 5시간 11분). 모델 로딩, audit 저장, 후속 gradient/loss 분석 및 rollout 재평가는 별도다.
- 단일 step 분석은 저장 단위를 뜻한다. 중간·후기 update를 분석하려면 해당 step까지 정상 학습해야 한다.

## 최종 checkpoint 분석과 단일 step 분석의 역할

사용자는 최종 gradient와 최종 파라미터 이동이 더 중요한지 질문했다. 최종 망각의 크기와 endpoint geometry를 확인하는 데에는 단일 step 재학습이 필수가 아니다. Δθ = θ₁ − θ₀에는 실제 AdamW 누적 이동이 이미 반영된다. AdamW라는 이유만으로 endpoint 분석을 배제해서는 안 된다.

현재 우선순위는 기존 checkpoint에서 gradient 방향·norm, 누적 displacement, 고정 probe 실제 loss 차이 및 기존 rollout success를 측정하는 것이다. 시작점 gradient에 대한 누적 displacement 투영은 보조적인 1차 근사이며, 전체 loss 차이나 인과 증명으로 해석하지 않는다. 단일 step audit는 시간별 국소 간섭이 추가 연구 질문으로 확정될 때 실행한다. 재학습은 아직 시작하지 않았다.

## 사용자 최신 범위 확정

가장 먼저 checkpoint 0의 Task 0 raw gradient와 checkpoint 1의 Task 0 raw gradient를 비교한다. 현재 실행 결과에서 이 두 gradient의 cosine, norm, noise 반복 변동을 우선 보고한다. 누적 displacement 분석 대기 작업을 중단했으며 AdamW 재학습은 시작하지 않았다.

## Task 0 raw gradient 관찰 완료

- checkpoint 0/1의 noise 평균 raw gradient cosine: 0.011290.
- 평균 gradient norm: 0.221063 → 3.929843.
- noise/time 반복별 cosine: [-0.0449782086103474, 0.0569258257158574, -0.025098467325669, 0.0352496183501366]. 부호가 반복마다 바뀌므로 안정적인 음의 정렬이라고 주장할 수 없다.
- mean-gradient norm / MC vector standard error: C0 2.805, C1 10.309. 두 값 모두 기존 low-signal 휴리스틱 기준(norm < 2×SE)에는 해당하지 않는다. 이 기준은 유의성 검정이 아니다.
- 보조 지표인 동일 Task 0 probe loss: 0.030212 → 0.649747.
- [상세 결과](outputs/analysis/libero_spatial/none/task0_to_1_adamw_20260927/RAW_GRADIENT_RESULTS.md).
- 원본 forward 일치, 동일 seed 재현성, batch-size gradient 안정성 검사를 통과했다. 누적 displacement 분석과 optimizer 재학습은 실행하지 않았다.

## Task 0~5 확장 실행

사용자 요청에 따라 기존 checkpoint 0~5에서 같은 task의 raw gradient 변화만 측정한다. 학습이 완료된 task j는 checkpoint j부터 checkpoint 5까지 평가한다(21개 task/checkpoint 조합 × noise/time 4회 = 84회 gradient 평가).

- 학습 직후 기준: cos(g_j(C_j), g_j(C_k)), j < k. 15개 비교.
- 인접 checkpoint 기준: cos(g_j(C_(k−1)), g_j(C_k)), j < k. 15개 비교.
- Task 5는 checkpoint 5에서 기준 gradient/norm/loss만 계산되며 이후 망각 비교는 범위에 없다.
- 원래 Task 0→1 실험과 동일한 seed=42, demo 20개, episode당 frame 4개, 4회 noise/time 반복, batch size 4 및 precision을 유지한다.
- 실행: `python -u scripts/run_raw_gradient_checkpoints.py --num-tasks 6 --repeats 4 --batch-size 4 --output outputs/analysis/libero_spatial/none/raw_gradients_task0_to_5_20260927`.
- 새 출력 디렉터리를 사용하며 optimizer update나 재학습은 수행하지 않는다.

## '최종 gradient' 범위 설명

현재 실험은 각 Task의 최종 checkpoint(모두 checkpoint-10000)에서만 Task별 raw gradient를 다시 계산한다. 학습 중간 optimizer step이나 중간 checkpoint를 평가하는 실험이 아니다. 84회는 학습 step 수가 아니라 21개 checkpoint/task 조합 × 4회 flow-matching noise/time 반복이다.

만약 최종 checkpoint 5에서의 누적 변화만 연구 질문이라면 각 Task 학습 직후와 checkpoint 5만 비교해도 된다. 이 경우 11개 고유 task/checkpoint 조합(6개 기준값과 기존 task 0~4의 최종값)으로 축소할 수 있다. checkpoint 0~5 사이의 변화 과정을 보려면 현재 범위를 유지한다. 사용자가 의미한 '최종'의 범위를 확인 중이며 현재 승인된 raw gradient 계산은 계속 진행한다.

사용자 확인: '각 Task의 최종 체크포인트 0~5 비교 — 현재 방식'을 선택했다. 21개 task/checkpoint 조합 × 4회 반복 범위를 유지한다.

## Task 0~5 실험 완료

- 21개 checkpoint/task 조합, noise/time 반복 4회, 총 84회 gradient 평가를 완료했다.
- 학습 직후 기준 15개 비교의 평균-gradient cosine은 −0.058114~0.343005다. 이 중 12개는 반복 간 부호가 바뀌었다. 안정적인 음의 정렬로 단정하지 않는다.
- 인접 checkpoint 기준 15개 비교 중 5개는 반복 간 부호가 바뀌었다. 학습 직후 기준과 인접 기준은 다른 연구 질문이다.
- 바로 다음 Task 학습 후 기존 Task gradient norm은 학습 직후 대비 17.78~43.49배다.
- 전체 action head의 개별 gradient 21개 중 Task 5/checkpoint 5만 기존 low-signal 휴리스틱(norm < 2×MC vector SE)에 해당한다(SNR 1.980). 이 gradient는 후속 checkpoint 비교에는 사용되지 않는다.
- Task 0→1 결과는 기존 실행과 cosine 및 양쪽 gradient norm이 동일하게 재현됐다. 원본 forward 일치, 동일 seed 재현, batch-size 안정성 검사를 통과했다.
- [전체 결과](outputs/analysis/libero_spatial/none/raw_gradients_task0_to_5_20260927/RESULTS.md), [요약 CSV](outputs/analysis/libero_spatial/none/raw_gradients_task0_to_5_20260927/raw_gradient_summary.csv), [전체 그림](outputs/analysis/libero_spatial/none/raw_gradients_task0_to_5_20260927/figures/01_raw_gradient_overview.png).
- 분석 코드, 환경 버전 및 SHA256을 결과 디렉터리에 보관했다. 재학습·단일 optimizer step 분석·누적 displacement 분석은 실행하지 않았다.
## Full-frame raw-gradient 실행 비용 추정 (2026-09-27)

- 범위: Task 0--5, checkpoint/task 조합 21개, noise/time 반복 4회, action horizon 16에서 가능한 모든 시작 frame.
- 데이터: 260 episodes, 26,886 valid frames.
- 총 sample 평가 수: 반복당 93,876개, 4회 합계 375,504개.
- 기존 pilot: 80 probes/task, 총 6,720 sample 평가의 gradient 단계가 약 22분 소요됨.
- 동일 RTX 4090 1장 기준 선형 환산: gradient 계산 약 20.6시간, VLM feature cache와 입출력을 포함해 약 20--26시간.
- 예상 cache 크기: 약 60--65 GB. 현재 여유 공간 132 GB에서는 가능하지만 중간 gradient 및 여유 공간을 포함해 최소 80 GB를 확보한다.
- 위 시간은 현 구현의 batch size 4와 반복 4회를 유지할 때의 추정치다. 반복 수를 수렴 기준으로 늘리면 거의 선형으로 증가한다.
## Cross-task raw-gradient conflict pilot (2026-09-27)

Conflict는 같은 pre-training checkpoint에서 old-task gradient와 incoming-task gradient를 비교한다.

\[
c_{j,k}=\cos\!\left(\bar g_j(C_{k-1}),\bar g_k(C_{k-1})\right),\qquad j<k.
\]

- Task 0--5의 15개 pair를 평가했다. 기존과 동일한 20 episodes/task, 4 temporal probes/episode, noise/time 반복 4회의 pilot probe bank를 사용했다.
- 전체 action head에서 14/15 pair는 4회 반복 모두 양수였다.
- 유일한 안정적 음수 pair는 checkpoint 2에서 old Task 1과 incoming Task 3이다: mean-gradient cosine -0.193595. 반복별 cosine은 -0.1932, -0.1680, -0.1617, -0.2308이며 episode bank A/B에서도 모두 음수였다.
- Task 0--1 conflict cosine은 +0.298170인데 Task 0 rollout success는 100 percentage points 하락했다.
- Task 2--3 conflict cosine은 +0.217804인데 Task 2 rollout success는 93.3333 percentage points 하락했다.
- 따라서 이 pilot은 raw-gradient conflict가 관찰된 망각 전반을 설명한다는 가설을 지지하지 않는다. Task 1--3의 국소 conflict는 관찰되지만, AdamW의 실제 update 원인으로 해석하지 않는다.
- 이 결과는 80-probe pilot이다. 최종 통계 주장은 모든 valid frame과 episode-cluster uncertainty로 재검증해야 한다.
- 결과: `outputs/analysis/libero_spatial/none/raw_conflict_task0_to_5_pilot_20260927/RESULTS.md`
