# Frozen-VLM Continual VLA: Catastrophic Forgetting 분석 계획

수정일: 2026-09-30

상태: 실험 환경·기록/평가 코드 및 데이터 없는 테스트를 구성했다. 실제 데이터/모델 기반 학습·평가는 아직 실행하지 않았다. 이후 논의는 별도 계획서를 늘리지 않고 이 문서에 반영한다.

## 1. 연구 목표와 이번 수정의 핵심

**목표는 VLM을 동결한 Continual VLA에서 action head의 gradient conflict, 동일 task에 대한 gradient drift, 기존 EWC Fisher map을 통해 catastrophic forgetting의 발생 원인을 분석하는 것이다.**

사용자가 말한 “catastrophic learning”은 문맥상 순차 학습 중 이전 task의 성능이 저하되는 catastrophic forgetting을 뜻하는 것으로 정리한다. 태스크 순서와 seed 반복은 이 분석을 검증하기 위한 실험 조건이지 연구 목표 자체가 아니다.

주요 질문은 다음과 같다.

1. 이전 task와 다음 task의 AdamW 보정 방향은 어디서, 얼마나 충돌하는가?
2. 같은 task의 보정 gradient가 다음 phase 학습을 거치며 얼마나 달라지는가?
3. 그 변화가 action head의 어떤 모듈·민감한 파라미터에 집중되고, 이전 task 성공률 하락과 어떻게 연결되는가?
4. 이런 관계가 task 순서와 학습 seed를 바꿔도, spatial·goal·object에서 반복되는가?

파라미터의 optimizer step 전후 차이는 주 분석 대상이 아니다. 기본 실험은 원인에 대한 가설과 관측 근거를 확보하는 단계이며, 상관관계만으로 특정 요인이 망각의 원인이라고 확정하지 않는다.

이번 수정에서는 이전 초안의 과도한 추가 설정을 제거한다.

- 학습 seed를 초기화·noise·augmentation·replay seed로 새로 분해하지 않는다. 기존 `--seed`를 사용한다.
- Fisher의 고정 subset, 128 samples × 2회, per-sample estimator, 별도 FP32 누적 변경을 기본안에서 제거한다.
- 500-step 누적 평균과 실제 파라미터 변화량의 동시 저장을 필수로 강제하지 않는다.
- EMA, best checkpoint, 새로운 CL 방법 비교는 기본 실험에 추가하지 않는다.
- 사용자 지정 사항 외에는 실제 사용하는 기존 학습·평가 진입점의 기본 설정을 유지한다.

## 2. 실험 설정

### 2.1 사용자 지정 사항

| 항목 | 설정 |
| --- | --- |
| 분석 모델 | VLM 동결, action head 학습·분석 |
| Suite | `libero_spatial`, `libero_goal`, `libero_object` |
| Task 범위 | 각 suite의 task ID 0~5, 총 6개만 학습하고 종료 |
| Task 순서 | 원래 순서와 여러 shuffled 순서; 순서용 seed 추가 |
| 반복 | 모든 순서에 기존 학습 seed 3개를 동일하게 적용 |
| 성능 집계 | 실행별 성공률 보존 후 순서별 3개 seed 평균·표준편차 |
| 학습 길이 | 기존 설정의 phase당 10,000 step |
| Gradient 기록 | 매 phase의 500, 1000, …, 10000 step에서 같은 방식으로 기록 |
| 분석 gradient | AdamW가 보정한 action-head 업데이트 방향 |
| Fisher | 기존 EWC `compute_fisher()`의 계산 방식을 그대로 사용 |
| 평가 | phase 최종 모델로 지금까지 학습한 task를 각각 30 rollout |
| 영상 | 기본 생성하지 않음; 요청한 checkpoint/task/episode만 별도 생성 |
| 시각화 | heatmap, 곡선·산점도, t-SNE 중심 |
| 저장 식별자 | suite, 순서 번호·seed, 학습 seed, phase, 원본 task, step |
| GPU | 할당된 물리 GPU 0만 사용; 다른 GPU fallback 금지 |

0~5는 같은 task 집합을 선택한 뒤 순서를 섞는 기준이다. 전체 10개를 섞어 앞 6개를 뽑아 task 구성까지 바꾸는 실험은 하지 않는다. 실제 canonical ID와 데이터 이름의 대응은 LIBERO 정의에 맞춰 manifest에 저장한다.

### 2.2 변경하지 않는 기존 기본값

현재 [sequential 학습 코드](scripts/gr00t_sequential_finetune.py)는 `tune_llm=False`, `tune_visual=False`, `tune_projector=True`, `tune_diffusion_model=True`이다. 여기서 projector는 코드상 action head 쪽 모듈이다. 실제 `requires_grad` 목록을 확인하고 그 범위에서 분석한다. VLM을 추가로 학습하거나 action-head projector를 임의로 동결하지 않는다.

- 모델: 기존 `nvidia/GR00T-N1.5-3B` 경로/사용 checkpoint.
- Optimizer: AdamW, LR `1e-4`, betas `(0.95, 0.999)`, eps `1e-8`, weight decay `1e-5`.
- Schedule: cosine, warmup ratio `0.05`; phase마다 기존 방식대로 재시작.
- 기존 BF16/TF32, gradient clipping, gradient accumulation, 데이터 전처리·증강, LoRA 설정을 유지한다.
- 학습 seed 기본값은 42이고, 평가 seed 기본값은 7이다. 평가 seed를 별도 제안값으로 교체하지 않는다.
- 학습 step의 의미와 종료 조건은 기존 Trainer 규칙을 유지한다. 진단을 위해 학습 스케줄을 재정의하지 않는다. update skip 발생 여부는 로그에 남겨 기록 시점의 상태를 구분한다.

**진입점별 기본값 차이를 숨기지 않는다.** [학습 명령 생성기](bash_scripts/generate_train_commands.sh)는 CL 방법 ER, batch 16을 기본으로 하고, Python dataclass 자체는 `cl_type=none`, batch 32가 기본이다. 현재 README의 bash 실행 경로를 기준으로 하면 ER/16을 유지한다. 실제 사용 명령에서 다른 값을 지정했다면 그 값을 따른다. Fisher를 분석한다는 이유로 학습 방법을 EWC로 바꾸지 않는다.

ER을 사용하는 경우 기존 memory·sampling 설정도 유지한다. 기본 경로에서는 현재 task 16 + replay 16이며, 첫 task에는 replay가 없다. `n_memories=1000`은 이전 task당 dataset item 수이다. ER의 실제 gradient는 current/replay 혼합 loss의 방향이므로 task별 진단과 구분해 기록한다.

## 3. Task 순서와 기존 학습 seed의 조합

### 3.1 두 seed의 역할

- `order_seed`: 선택한 6개 task의 순서만 생성하는 새 설정.
- `seed`: 현재 코드에 이미 있는 학습 seed. dataloader shuffle에도 사용되므로 사용자가 말한 “배치 섞는 seed”는 이 설정을 그대로 활용한다.

기존 `seed`는 [sampler](gr00t/experiment/trainer.py)뿐 아니라 [runner의 set_seed](gr00t/experiment/runner.py)와 replay 선택 등에도 연결된다. 따라서 결과는 기존 학습 seed 반복으로 보고하며, 배치 순서만의 인과 효과를 분리했다고 주장하지 않는다. 이를 이유로 새로운 난수 분리 체계를 도입하지 않는다.

순서 생성에는 독립 `random.Random(order_seed)`를 사용해 학습 RNG를 건드리지 않는다. seed 값과 실제 순서 배열을 모두 저장한다.

### 3.2 실행 조합

shuffle 수와 구체적인 seed 숫자는 아직 사용자 고정값이 아니므로 다음을 시작안으로 둔다.

| 순서 조건 | order seed | 원본 task 순서 | 기존 학습 seed |
| --- | --- | --- | --- |
| order00: canonical | null | 0, 1, 2, 3, 4, 5 | 42, 43, 44 |
| order01: shuffle A | 0 | 4, 2, 1, 0, 5, 3 | 42, 43, 44 |
| order02: shuffle B | 1 | 2, 3, 5, 0, 4, 1 | 42, 43, 44 |
| order03: shuffle C | 2 | 2, 3, 1, 4, 5, 0 | 42, 43, 44 |

각 `(suite, order, seed)`는 독립적인 6-task 학습 경로이다. 동일한 시작 checkpoint를 사용하되, 경로 안에서는 직전 phase의 최종 가중치를 이어받는다. seed별 모델을 평균해서 다음 task에 넘기지 않는다.

`phase`는 해당 실행의 학습 위치 0~5이고, `canonical_task_id`는 원본 task ID이다. 예를 들어 order01의 phase00은 원본 task04이다. 학습·평가·gradient·Fisher 모두 이 둘을 구분한다.

## 4. AdamW 초기화와 gradient 기록

### 4.1 Phase 전환은 기존 방식 유지

현재 코드는 이전 phase의 `checkpoint-{max_steps}`를 다음 phase의 base model로 불러오고, 기본 `resume=False`에서 새 Trainer/optimizer를 구성한다. **모델 가중치는 이어받고 AdamW의 moment·step 및 LR schedule은 초기화하는 기존 프로토콜을 그대로 유지한다.**

가중치를 이어받는다고 수학적으로 optimizer 초기화가 필수인 것은 아니지만, 이번 실험에서는 원래 코드의 선택을 바꿀 이유가 없다. optimizer 초기화는 수정 대상이 아니라 분석 결과에 명시할 실험 조건이다. 같은 phase의 중단 재개와 다음 phase의 새 학습은 구분한다.

### 4.2 기록할 보정 gradient

기록 대상은 raw `.grad`나 단순 파라미터 차이가 아니라, 해당 학습 step에서 AdamW의 moment와 bias correction을 거친 방향이다.

```text
g_t = backward 및 기존 accumulation/clipping을 거쳐 optimizer에 입력된 gradient
m_t = β1 m_(t−1) + (1−β1) g_t
v_t = β2 v_(t−1) + (1−β2) g_t²
u_t = [m_t / (1−β1^s)] / [sqrt(v_t / (1−β2^s)) + ε]

d_t = u_t + λ_group × θ_before
실제 업데이트 ≈ −LR_group,t × d_t
```

주 분석 벡터는 **AdamW 보정 방향 `u_t`**이다. Weight decay는 AdamW에서 별도 적용되므로 loss 관련 방향과 섞지 않고 LR·decay·parameter group을 metadata에 남긴다. decay 포함 방향 `d_t`나 실제 변화량은 필요한 경우의 보조 분석이지 필수 주 분석이 아니다. [AdamW 공식 정의](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)

`s`는 파라미터별 optimizer step이다. `grad=None`으로 건너뛴 좌표는 적용 방향 0 및 비활성 상태로 표시한다. gradient 값이 0인 tensor와 `None`은 다르다. step 이후 `.grad`만 읽어 보정 방향으로 오인하지 않는다.

실제 step의 `u_t`는 그 step 직전 모델과 갱신된 moment에 대응한다. 모델 checkpoint는 통상 step 이후 상태이므로 metadata에 전후 시점을 구분한다.

### 4.3 500-step 기록의 의미

이번 설명의 핵심은 **모든 phase에서 동일한 phase-local 간격으로 기록해 비교한다**는 것이다. 사용자의 “누적과 해당 시점 중 어느 쪽이 실험적으로 적합한가?”에 대해서는, **이 연구 목표에는 500·1000·…·10000 step의 snapshot을 주 분석으로 권장**한다. 이전 초안처럼 500개의 방향을 반드시 누적해야 한다고 단정하지 않는다.

| 방식 | 답하는 질문 | 이 실험에서의 용도 |
| --- | --- | --- |
| 500-step 간격 snapshot | 이 학습 상태에서 어느 방향으로 학습하는가? 같은 task의 방향이 어떻게 달라졌는가? | 주 기록; 같은 상태의 task별 진단과 결합 |
| 직전 500-step 방향 평균 | 이 구간 전체에서 평균적으로 어느 방향을 향했는가? | 학습 궤적의 보조 요약 |

누적 평균은 배치 변동을 완화할 수 있지만, 구간 안에서 모델 파라미터와 AdamW state가 계속 바뀐다. 따라서 한 checkpoint의 task conflict나 같은 task의 시점 간 drift와는 다른 양이다. 예를 들어 앞부분과 뒷부분의 방향이 반대이면 구간 평균이 작아져 중간의 강한 충돌·방향 전환을 숨길 수 있다. 또한 평균 방향끼리의 cosine과 각 시점 cosine의 평균은 같지 않다.

snapshot도 단일 배치의 영향을 받으므로 완벽한 task 대표값은 아니다. 다만 저장하는 `u_t`는 raw gradient가 아니라 이미 Adam의 moment를 반영한다. 진단의 배치 변동이 너무 크다면 **같은 checkpoint와 동일한 optimizer state에서 여러 분석 배치를 계산해 지표의 평균·편차를 보는 것**이 목적에 더 직접적이다. 이때 배치별 대응 보정 방향과 cosine/drift를 먼저 계산한 뒤 지표를 평균한다. raw gradient를 먼저 평균한 뒤 AdamW 보정을 적용하는 방식은 비선형성 때문에 다른 추정량이므로 혼용하지 않는다. 이는 모델이 바뀌는 500번의 학습 step을 평균하는 것과 다르며, 기본 학습 batch나 Fisher 계산을 바꾸지 않는다.

구간 누적 평균을 나중에 추가한다면 snapshot과 별도 키로 저장한다. 매 step 계측·누적이 추가되므로 기본안의 필수 비용으로 넣지 않는다. 500-step 기록 자체가 그 사이의 모든 순간적 충돌을 포착한다는 보장은 없으며, 경계 부근 변화가 중요해지면 필요한 구간만 더 촘촘히 보는 후속 진단을 검토한다.

- 정상 완주 시 phase당 20개 기록 시점, 6개 phase당 120개 시점.
- 매 기록 시점에 actual training 방향, task별 진단 방향과 지표를 구분해 저장한다.
- gradient 기록 파일은 모델 checkpoint와 별도로 보존한다. 500 step마다 전체 모델을 복제한다는 뜻으로 확대하지 않는다.
- `save_total_limit=8` 등의 모델 checkpoint 정리 정책이 gradient 기록을 삭제하지 않게 한다.
- 모든 파일에 suite/order/seed/phase/task/phase-local step 및 optimizer state 기준을 남긴다.

## 5. Gradient conflict와 동일 task drift

### 5.1 서로 다른 세 가지 비교를 구분

| 분석 | 비교 대상 | 의미 |
| --- | --- | --- |
| 관측된 phase 간 방향 | 이전·다음 phase에서 실제 사용한 `u_t` | 실제 학습 방향의 변화; 모델·optimizer·배치 차이를 모두 포함 |
| Task 간 conflict | 한 모델·동일 AdamW state에서 이전 task와 현재 task의 보정 방향 | 같은 상태에서 두 task가 요구하는 optimizer-aware 방향의 불일치 |
| 동일 task drift | 동일 task를 이전·다음 phase의 모델에서 다시 계산한 보정 방향 | 다른 task 학습 이후 같은 task에 대한 방향이 얼마나 달라졌는지 |

이전 phase의 task A gradient와 다음 phase의 task B gradient만 비교해서는 **task A 자체의 drift**를 알 수 없다. 다음 phase 모델에서도 A의 loss를 다시 미분해야 한다.

### 5.2 분석 전용 task별 gradient

학습은 기존 loss와 배치 구성으로 그대로 수행한다. 추가 backward는 분석만을 위한 것이며 optimizer step을 적용하거나 replay·학습 loss를 바꾸지 않는다.

각 기록 시점의 모델과 optimizer state를 기준으로 task별 gradient를 각각 계산한다. 동일 state의 독립 사본에 각 gradient가 들어간다고 가정해 **후보 AdamW 보정 방향**을 산출한다. 이전 task를 먼저 계산한 결과로 optimizer state를 바꾼 뒤 현재 task를 계산하면 안 된다.

```text
q_j(θ, S) = 같은 모델 θ와 AdamW state S에서 task j의 gradient로
            가상의 다음 AdamW 계산을 했을 때 얻는 보정 방향
```

`q_j`는 해당 상태에서 그 task를 학습한다면 적용될 후보 방향이며, 실제 학습에 적용된 `u_t`와 같다고 표기하지 않는다. 두 종류를 `train_actual`과 `task_diagnostic`으로 분리한다. task별 gradient의 accumulation/clipping 처리는 사용된 기준을 기록한다.

같은 task의 drift 비교에는 **같은 분석 배치와 noise/time·증강 조건을 대응**시킨다. 이는 Fisher 계산에 고정 subset을 도입하는 변경이 아니라, 입력 차이를 모델 변화로 오해하지 않기 위한 gradient 비교 조건이다. sample 수 128 등 새 수치를 강제하지 않고 기존 batch 크기를 활용하는 최소 진단부터 비용을 확인한다. 단일 배치 결과는 해당 배치의 진단임을 밝히고 task 전체를 대표한다고 단정하지 않는다.

추가 진단은 모델 mode, 기존 gradient, RNG, optimizer state를 바꾸지 않도록 격리한다. flow-matching loss는 eval mode에서도 noise/time을 샘플링하므로 mode만 같게 하는 것으로 충분하지 않다.

### 5.3 Conflict 분석

기본 비교 대상은 인접 phase의 이전 task A와 현재 task B이다. 같은 checkpoint의 `θ, S`에서 다음을 계산한다.

```text
conflict_similarity(A,B; θ,S) = cosine(q_A(θ,S), q_B(θ,S))
```

음의 cosine은 보정 방향의 충돌 신호로 해석한다. 두 후보 방향에는 공통의 기존 momentum도 포함되므로, 양의 cosine이라고 raw task-gradient 충돌이나 망각 위험이 없다고 결론 내리지 않는다. 전체 action head와 주요 모듈별로 기록하고, phase-local step에 따른 곡선과 heatmap을 만든다. 실제 학습 방향의 phase 간 비교도 함께 표시하되 같은 상태의 task-conflict 분석과 혼동하지 않는다.

일반적인 raw-gradient conflict 정의는 같은 파라미터에서 task별 gradient를 비교한다. 여기서는 AdamW 보정 방향으로 확장한 진단이므로, 음의 cosine만으로 이전 task loss 증가나 망각을 보장한다고 주장하지 않는다. [Gradient Surgery 논문](https://proceedings.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)

### 5.4 동일 task drift 분석

task A를 phase i에서 학습했다면 A의 분석 배치를 phase i와 phase i+1 모델에서 모두 계산한다. 이전 phase 마지막 기록을 기준으로 다음 phase의 500·1000·…·10000 step을 추적하고, 동일 phase-local step끼리의 비교도 분리해서 볼 수 있다.

```text
direction_drift_A = 1 − cosine(q_A(θ_before,S_before), q_A(θ_after,S_after))
relative_drift_A  = ||q_A_after − q_A_before|| / (||q_A_before|| + ε)
```

각 phase의 실제 AdamW state를 사용한 이 지표는 **모델과 optimizer 상태를 함께 반영한 보정 방향 drift**이다. phase별 초기화도 이 프로토콜의 일부이다. 모델 변화만 분리하려면 같은 reference AdamW state를 두 모델에 적용하는 보조 비교가 필요하지만, 이를 위해 본 학습의 optimizer 초기화 정책을 바꾸지 않는다.

norm이 0인 방향의 cosine은 정의 불가로 남기며 거의 0인 경우도 불안정 표시를 한다. seed별 유사도·drift를 먼저 계산한 뒤 평균·표준편차를 낸다. seed별 gradient를 먼저 평균한 뒤 cosine을 계산하지 않는다.

## 6. Fisher: 기존 EWC 계산 그대로 사용

### 6.1 계산 방식 유지

[기존 compute_fisher](scripts/gr00t_sequential_finetune.py)는 이미 독립 함수이다. 현재 EWC 학습일 때만 호출되지만, 이 호출 조건과 결과 저장만 분석용으로 분리하면 다른 CL 학습에서도 사용할 수 있다. EWC regularization을 켤 필요는 없다.

**계산 내부의 기존 방식은 바꾸지 않는다.**

- 해당 task의 전체 dataset.
- 기존 `DefaultDataCollator`, batch size 8, `shuffle=True`.
- `model.eval()`에서 `model(batch)["loss"]` 역전파.
- `requires_grad` 파라미터별 loss gradient 제곱을 누적하고 batch 수로 나눔.
- 기존 dtype·mask·reduction·noise sampling을 유지.
- 고정 probe subset, 추가 noise 반복, per-sample gradient estimator로 교체하지 않음.

```text
F_j[p] = (1 / number_of_batches) × Σ_batch (∂loss_batch / ∂θ[p])²
```

이 값은 기존 코드에서 Fisher라고 부르는 **batch-gradient 제곱 평균 기반 대각 근사**이다. 전체 P×P 행렬이나 sample별 제곱 평균과는 다르다는 사실만 정확히 기록한다. loss가 flow-matching masked MSE이므로 엄밀한 likelihood Fisher와 동일하다고 단정하지 않는다. 이 한계가 기존 방식을 반드시 변경해야 한다는 뜻은 아니다.

또한 여기서는 **raw loss gradient로 계산하는 기존 Fisher**를 그대로 쓴다. 분석용 AdamW 보정 방향을 제곱해 Fisher로 대체하지 않는다.

### 6.2 호출·저장에 필요한 최소 처리

- 각 phase 최종 모델에서 현재 task의 Fisher map을 계산·저장한다.
- ER에서는 current/replay 혼합 loss가 아니라 기존 EWC가 받는 형태의 해당 task dataset을 넘긴다.
- task별 map을 저장한 뒤 필요한 후속 처리를 한다. EWC 누적 dictionary의 in-place 변경과 공유되지 않도록 독립 사본을 보존한다.
- 분석 호출의 train/eval mode·gradient·RNG 부작용이 다음 학습에 남지 않게 외부 호출부에서 격리한다. 계산 estimator 자체는 변경하지 않는다.
- dataset, batch 수·크기, 실제 실행 seed/RNG 기준, parameter 이름·shape, checkpoint ID를 기록한다.
- batch shuffle·증강·noise에 따른 변동성은 기존 방식의 특성으로 인정하고 결과 해석에 명시한다.

이전에 고정 probe를 제안한 목적은 비용과 비교 변동성을 줄이기 위해서였다. **기존 구현 재사용의 필수 조건은 아니므로 이번 기본 실험에서는 적용하지 않는다.** 전체 dataset 계산 비용은 실제 데이터 크기로 측정한다.

### 6.3 분석 지표

action head 모듈별 Fisher 평균·합·분포와 상위 민감 좌표를 heatmap으로 보여준다. 모듈 크기에 따라 합이 달라지므로 평균도 함께 본다. Fisher가 큰 모듈에서 conflict·drift와 이전 task 성공률 저하가 함께 나타나는지 분석한다.

필요하면 이전 task 직후 map을 고정한 `Σ_p F_j[p](θ_later[p]−θ_j[p])²`를 보조 지표로 계산한다. 이는 Fisher 가중 파라미터 이동량이지 주 목표인 gradient drift의 대체물이 아니며, 정확한 손실 증가량으로 해석하지 않는다.

## 7. 성능 평가: 30 rollout, 영상 없음

### 7.1 평가 규칙

각 phase가 끝나면 최종 checkpoint로 지금까지 학습한 task를 각각 30회 평가한다. 예를 들어 phase 2에서는 그 순서의 첫 3개 task를 평가한다.

- 기존 평가의 `num_trials_per_task`를 50에서 **30**으로 변경한다.
- 성공 여부·episode ID·초기 상태 ID·평가 seed·종료 원인 등을 JSONL/CSV로 보존한다.
- 기본 영상 인코딩·파일 저장·영상용 frame buffer와 debug PNG 저장은 끈다.
- 정책 입력에 필요한 simulator camera observation/rendering은 유지한다. “영상 미저장”은 “시각 입력 제거”가 아니다.
- 평가 seed 7과 기존 inference·timeout·전처리 기본값은 유지한다. 같은 canonical task의 평가 초기 상태 목록을 맞춘다.
- 영상이 필요하면 요청된 run/checkpoint/task/episode만 별도 실행·저장하고 원래 정량 평가 결과를 덮어쓰지 않는다.

현재 평가 코드는 영상 개수로 skip/resume을 판단하고 `>40` 조건도 사용한다. **30회·영상 없음 조건에서는 episode 결과와 완료 표식으로 바꿔야 한다.** 이 부분은 사용자 설정을 적용하기 위한 필수 변경이다.

### 7.2 지표와 평균 단위

`A[i,j]`는 학습 위치 i 종료 모델의, 위치 j에서 배운 task 성공률이다. task 이름과 canonical ID를 함께 표시한다.

```text
최종 평균 성공률 = mean_j A[5,j], j=0..5
forgetting(j) = max_(k=j..5) A[k,j] − A[5,j], j=0..4
최종 평균 forgetting = mean_j forgetting(j), j=0..4
BWT = mean_j (A[5,j] − A[j,j]), j=0..4
```

각 순서에서 기존 학습 seed 3개로 계산한 성공률 행렬의 평균·표준편차와 개별 값을 저장한다. forgetting은 실행별로 먼저 계산한 뒤 집계한다. 30 rollout은 한 모델의 평가 표본이고 독립 학습 seed 30개가 아니다.

canonical 순서는 reference로 별도 표시하고 shuffled 순서 평균에 섞지 않는다. task별 순서 비교는 같은 position 숫자가 아니라 같은 canonical task를 맞춘다. 학습 이후 경과 phase도 기록해 오래된 task의 자연스러운 차이를 통제해 해석한다.

## 8. 마지막 checkpoint를 평가하는 이유와 EMA/best

### 8.1 코드에서 확인한 사실

- sequential loop는 직전 phase의 `checkpoint-{max_steps}`를 다음 phase base model로 읽는다.
- 평가 명령 생성기는 기본 `CKPT_STEP=$STEPS`로 마지막 checkpoint를 선택한다.
- 학습은 `do_eval=False`이며, 해당 runner 경로에 validation dataset이나 best-model 선택 설정이 없다.
- 프로젝트 학습 경로에서 EMA 가중치 추적·교체 구현도 확인되지 않았다.

따라서 현재 기본 결과는 EMA나 검증 성능으로 고른 best가 아니라 마지막 가중치의 평가이다. 논문 저자가 왜 이 설계를 선택했는지까지 코드만으로 단정하지 않는다.

### 8.2 이번 분석에서 유지할 근거

마지막 checkpoint는 **정해진 phase 학습을 마친 실제 모델이고 다음 phase로 넘기는 모델**이다. 이 상태의 gradient/Fisher와 이전 task 성공률을 연결하면 분석 대상이 일치한다.

EMA는 학습 중 가중치를 지수 이동 평균한 별도 모델이다. 좋은 성능을 보일 가능성은 있지만 항상 우월한 것은 아니며, decay·시작 시점·phase 간 유지/초기화 같은 조건이 추가된다. 마지막 모델의 gradient를 측정하고 EMA 모델을 평가하면 분석 대상도 달라진다. [PyTorch의 EMA 설명](https://docs.pytorch.org/docs/2.14/optim.html#weight-averaging-swa-and-ema)

best checkpoint는 “무엇의 최고인가”를 먼저 정해야 한다. 현재 task validation loss, 현재 task 성공률, seen-task 평균 성공률은 서로 다른 모델을 선택할 수 있다. 선택용 검증 평가도 추가로 필요하며, 최종 보고용 rollout 결과로 best를 고른 뒤 같은 결과를 그대로 보고하면 선택 편향이 생긴다.

**이번 기본 실험은 마지막 checkpoint를 유지한다.** EMA/best가 잘못된 방법이어서 제외하는 것이 아니라, 기존 프로토콜의 실제 망각을 분석하려는 목적에 맞추는 것이다. 추후 사용자가 요청하면 별도 평가 조건으로 비교한다.

## 9. 시각화와 고차원 처리

### 9.1 주요 산출물

| 시각화 | 내용 |
| --- | --- |
| 성공률 heatmap | phase × 평가 task; 순서별 seed 평균과 편차 |
| Conflict heatmap | task pair/phase × 기록 step; action-head 전체 및 모듈별 |
| Drift 그래프 | 동일 task의 다음 phase 진행에 따른 방향·크기 변화 |
| Fisher heatmap | phase/task × action-head 모듈의 민감도 |
| 연관성 그래프 | conflict·drift·Fisher 통계와 이전 task 성공률 하락 |
| t-SNE | 보정 gradient 분포를 task·phase별로 표시; 순서·seed 식별 |

### 9.2 차원과 해석

먼저 실제 분석할 action-head 학습 파라미터 수·모듈별 차원을 측정한다. 모델 이름의 3B를 그대로 gradient 차원으로 사용하지 않는다. 공통 parameter 순서·shape·활성 여부를 manifest로 저장한다.

저장량이 감당 가능하면 원본 방향을 보존한다. 크다면 공통 PCA 또는 검증한 선형 압축을 검토하되, 4096차원 같은 값을 미리 강제하지 않는다. PCA가 작은 분산을 제거한다고 그것이 반드시 노이즈인 것은 아니다.

t-SNE는 시각화용이며 conflict·drift 수치를 계산하는 공간으로 사용하지 않는다. 비교할 점들은 같은 전처리와 공동 embedding 안에 넣고, 따로 fit한 t-SNE 좌표를 phase 간 직접 비교하지 않는다. 입력 정규화·PCA 여부·random state·perplexity를 기록하고 표본 수에 맞게 설정한다. 군집 간 거리만으로 망각의 원인을 결론 내리지 않는다. [t-SNE 문서](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

정량 cosine·norm은 가능한 원래 좌표에서 계산한다. 압축을 쓴 경우 일부 원본으로 왜곡을 확인하고 근사 지표임을 명시한다.

## 10. 체크포인트와 파일 명명

### 10.1 식별 규칙

공통 prefix에 다음을 넣는다.

```text
{suite}__{method}__order{index}__oseed{value-or-na}__seed{training_seed}
__phase{position}__task{canonical_id}__step{phase_local_step}
```

예시: `libero_spatial__ER__order01__oseed0__seed42__phase00__task04__step00500__train_actual.pt`

`order01`은 실험 조건 번호이고 `oseed0`은 permutation 생성 seed이다. `seed42`는 기존 학습 seed이다. 둘을 단순히 “seed 1”처럼 모호하게 표기하지 않는다.

### 10.2 저장 구조

```text
output/continual_analysis/
  libero_spatial__ER__order01__oseed0__seed42__attempt01/
    training_args.json
    task_order.json
    parameter_manifest.json
    environment.json
    task_0/
      checkpoint-10000/       # 기존 로더와 호환; manifest에서 phase00=task04
      gradients/
        libero_spatial__ER__order01__oseed0__seed42__phase00__task04__step00500__train_actual.pt
        ...__step00500__evaltask04__batch00__task_diagnostic.pt
        ...__step01000__train_actual.pt
        ...
      fisher/
        ...__phase00__task04__step10000__ewc_fisher.pt
        ...__phase00__task04__step10000__ewc_fisher.json
      evaluation/
        canonical_task_04/
          ...__phase00__task04__step10000__evaltask04__episodes.jsonl
          ...__phase00__task04__step10000__evaltask04__metrics.json
    ...
    task_5/
  analysis_attempt01/
    per_run_summary.csv
    success_cells.csv
    per_order_seed_mean_std.csv
    gradient_metrics.csv
    fisher_by_module.csv
    *.png
    tsne_settings.json        # optional --tsne
    tsne_points.csv
```

위 `...` 파일명은 설명용 축약이며 실제로는 공통 prefix 전체를 쓴다. 다른 task를 진단·평가한 파일에는 학습 중 task와 `evaltask`를 별도 기록한다. 모델 checkpoint는 상위 run 경로로 suite/order/seed를 식별하며 기존 `task_i/checkpoint-10000` 연결을 보존한다.

기존 평가 경로는 학습 seed가 빠질 수 있으므로 반드시 run ID를 포함한다. 임의 순서를 파일명 정규식으로 추정하지 않고 manifest로 읽는다. Git commit·로컬 변경 hash·패키지/모델/데이터 버전·GPU UUID를 기록하고 재실행은 attempt ID를 늘려 결과를 덮어쓰지 않는다.

## 11. 규모와 최소 구현 순서

shuffle 3개를 사용하는 시작안이면 `3 suites × 4 orders × 3 training seeds = 36`회 전체 학습이다. shuffle 수는 제안값이며 확정 시 이 표를 함께 갱신한다.

| 항목 | 전체 학습 1회 | 시작안 36회 |
| --- | --- | --- |
| 학습 phase | 6 | 216 |
| 설정상 학습 step | 60,000 | 2,160,000 |
| Gradient 기록 시점 | 120 | 4,320 |
| Phase 최종 checkpoint | 6 | 216 |
| 평가 task/checkpoint 조합 | 21 | 756 |
| Rollout: 조합당 30회 | 630 | 22,680 |
| Fisher: 전체 task dataset 계산 | 6회 | 216회 |

기록 시점 하나에 actual gradient와 task별 진단 여러 개가 들어갈 수 있으므로 4,320시점이 곧 벡터 4,320개라는 뜻은 아니다. Fisher sample 수·역전파 비용은 기존 전체 dataset 크기에 따라 달라지며 임의의 128×2 예산을 쓰지 않는다.

구현은 아래 범위로 제한한다.

1. 기존 진입점·실행 설정을 보존하고, GPU 0과 VLM 동결/action-head 학습 범위를 확인한다.
2. order seed·6-task 종료·기존 seed 3회 조합·manifest와 명확한 저장 이름을 추가한다.
3. 500-step 보정 방향 기록과 task별 conflict/drift 진단을 추가한다. 진단이 실제 학습 상태를 바꾸지 않는지 작은 테스트로 확인한다.
4. 기존 Fisher 함수를 별도 진단으로 호출·저장한다. 추정 방식은 바꾸지 않는다.
5. 30 rollout·영상 없음 및 episode 결과 기반 재개를 적용한다.
6. spatial의 순서 하나·학습 seed 하나로 짧은 기능 검증 후 완주 비용을 측정한다.
7. 세 suite와 모든 order/seed 조합에 같은 프로토콜을 적용하고, GPU 0에서 순차 실행한다.
8. 실행별 지표를 보존한 뒤 seed 평균·표준편차와 heatmap/graph/t-SNE를 생성한다.

## 12. 구현 상태 및 환경 구성 (2026-09-30)

사용자의 “데이터는 나중에 준비하고 실험 환경부터 구성” 요청에 따라 본 계획의 실행 기반을 추가했다. **데이터·모델 가중치 다운로드 및 실제 학습은 실행하지 않았다.** 상세 명령은 [EXPERIMENT_SETUP.md](EXPERIMENT_SETUP.md)에 있다.

- 기본 설정: [configs/continual_analysis.json](configs/continual_analysis.json). 표의 시작안인 36 runs를 생성하며 숫자는 config에서 변경할 수 있다. canonical ID 0~5의 집합을 유지한다.
- 새 진입점: [scripts/continual_experiment.py](scripts/continual_experiment.py). `prepare`/ `doctor`는 데이터 없이 동작하고 `run`/ `evaluate`는 `--execute` 없이는 명령만 출력한다.
- 기존 sequential 학습에 manifest 기반 기록을 opt-in으로 연결했다. 기존 bash/Python의 기본값 차이를 임의로 통일하거나 학습 방법을 EWC로 변경하지 않았다.
- [continual_diagnostics.py](gr00t/experiment/continual_diagnostics.py): 500-step 실제 보정 방향, 같은 상태에서의 task별 후보 방향, 인접 phase conflict와 같은 task drift, parameter manifest, 원 좌표 전체/모듈별 지표.
- 진단은 기본 task당 1 batch이며 배치 크기는 기존 학습 설정을 따른다. sample indices 및 task별 분석 seed를 고정하여 같은 task의 noise/증강을 대응시킨다. 이는 Fisher를 fixed-probe estimator로 바꾸는 것이 아니다. 모델 RNG·gradient·train/eval flag·buffer를 복원한다.
- Fisher 함수 본문은 원본과 동일하다. 전체 current-task dataset, batch 8, shuffle, batch gradient 제곱 평균을 그대로 사용한다. 저장용 map은 EWC 누적 map과 독립 사본이다.
- 평가 전용 [scripts/eval_continual_libero.py](scripts/eval_continual_libero.py)를 추가했다. 기존 `eval_policy_libero.py`의 다른 suite용 동작은 보존하며 이번 실험은 새 manifest 경로로 실행한다.
- 30 rollout, 영상/debug PNG 없음, episode JSONL 및 완료 상태 기반 평가 재개를 적용한다. 평가 base seed 7을 유지하되 중단 재개의 재현성을 위해 episode별 policy RNG를 `7 + canonical_task_id × 1000003 + episode_id`로 대응시킨다. 모든 순서·학습 seed에 동일 규칙이며 기존 단일 연속 RNG stream과의 차이를 기록한다.
- object canonical ID 0~5는 공식 LIBERO 기준 alphabet_soup → cream_cheese → salad_dressing → bbq_sauce → ketchup → tomato_sauce이다. 기존 bash 파일명 정렬 순서와 다르므로 manifest에서 명시하고 공식 catalog와 검증한다.
- 원본 벡터를 보존하며 시작 시 실제 차원과 남은 gradient/Fisher 저장량 하한을 계산한다. 6-phase run당 actual 120개 + task diagnostic 220개 + Fisher 6개로, FP32 기준 약 `346 × 4P bytes`가 필요하다. checkpoint/임시 파일/다른 run 공간은 별도다. 공간이 부족하면 중단하며 임의 압축은 하지 않는다.
- [continual_analysis.py](gr00t/experiment/continual_analysis.py): 실행별 성공률·forgetting, 순서별 3 seed 평균/sample SD, success heatmap/curve, conflict/drift CSV·곡선, Fisher 모듈별 CSV·heatmap. 누락/error는 실패 0으로 대체하지 않고, 3개 seed 미완료 조건은 최종 평균에서 제외한다.
- optional t-SNE는 공통 CountSketch 512 → L2 정규화 → 공동 PCA → 공동 t-SNE의 **offline 시각화 근사**이다. 전처리 설정과 phase/order/seed label을 저장한다. 원본 저장이나 정량 cosine/drift의 차원을 줄이지 않으며 “압축=노이즈 제거”로 해석하지 않는다.
- GPU 0만 사용한다. 현재 GPU는 RTX 4090 24GB이다. 프로젝트 conda에 simulator/analysis 및 GL/EGL 의존성을 추가했고, 시스템 드라이버는 변경하지 않았다. 환경 활성화는 프로젝트 루트의 `source .envrc`이다.
- CPU 단위 테스트 14개와 GPU 0의 EGL 64×64 렌더링/초기 상태 설정/dummy action 1 step 검증을 통과했다. 실제 GR00T batch 16의 VRAM, 데이터 전처리, full Fisher 비용, 전체 학습·30 rollout은 데이터/모델 준비 후 pilot으로 검증해야 한다.
- 현재 자동 재개는 **평가 episode**에 해당한다. 중단된 전체 학습 chain을 자동 재개하지 않으며 기존 phase 디렉터리가 있으면 덮어쓰지 않는다. 재학습은 새 attempt를 사용한다.
- EMA/best 실험, 새 CL 방법, gradient/Fisher와 망각 사이의 인과 주장 및 심화 상관 분석은 이번 환경 구축에서 추가하지 않는다.


## 13. 논문 공식 저장소 및 다운로드 기준 확인 (2026-09-30)

논문 코드의 기준 출처는 사용자가 지정한 [Continual-VLAs/continual-GR00T](https://github.com/Continual-VLAs/continual-GR00T)이다. 사용자 fork는 작업 저장소로 유지한다. 확인 시점의 공식 HEAD와 로컬 HEAD는 모두 `4bfd66ff44a7fbf17dffa86e7c26ae4e0a4565c0`이다. 로컬 수정분은 이 기준 위에 추가된 것이며, 별도 clone/pull/merge 또는 origin 교체는 하지 않았다.

추가 다운로드의 기준은 공식 [설치 안내](https://github.com/Continual-VLAs/continual-GR00T/blob/4bfd66ff44a7fbf17dffa86e7c26ae4e0a4565c0/README_GR00T.md)와 [의존성 정의](https://github.com/Continual-VLAs/continual-GR00T/blob/4bfd66ff44a7fbf17dffa86e7c26ae4e0a4565c0/pyproject.toml)이다.

- 필요한 초기 가중치는 [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B)이며, 이번에는 다운로드하지 않았다. 실제 사용 시 모델 revision/파일 출처를 고정한다. 기본 로더는 저장소에 포함된 Eagle processor/config와 GR00T checkpoint를 사용하므로 별도 Eagle 가중치를 임의로 추가 다운로드하지 않는다.
- 필요한 데이터는 [공식 LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)의 spatial/goal/object에 해당하는 demonstration이며, 사용자가 나중에 준비한다. README_GR00T의 PhysicalAI 예제 데이터와 RoboCasa/GR-1 환경은 이번 LIBERO 실험의 필수 자료가 아니다.
- 설치된 LIBERO simulator/assets/init states는 demonstration dataset과 구분한다. TensorRT/Jetson 배포 구성과 Slurm job array는 이번 GPU 0 단일 PyTorch 실험에 적용하지 않는다.
- 저장소의 `convert_hdf5_to_lerobot.py`는 RoboCasa 관측/action과 `ep_meta["lang"]`을 전제로 하므로 raw LIBERO HDF5를 그대로 넣어 호환된다고 보지 않는다. 실제 다운로드한 데이터의 schema를 확인한 후 LIBERO→GR00T modality 매핑을 검증한다.

### 13.1 대조 과정에서 수정한 data config 불일치

이전 환경 구성에서 새 manifest의 `data_config`를 `libero90`으로 둔 것은 오류였다. 공식 학습 생성기의 `libero`와 현재 rollout adapter는 `video.front_view` / `video.left_wrist_view`, `state.eef_*`, `action.eef_*_delta`를 사용한다. 반면 `libero90`은 `video.image` / `video.wrist_image`, 축별 `state.x` / `action.x` 등을 사용하므로 서로 바꿔 쓸 수 없다.

새 config와 평가 기본값을 `libero`로 맞췄고, 지원되지 않는 schema는 모델 로딩 전에 거부한다. **활성 실행 설정은 `experiment_index__attempt02.json`의 36 runs**이다. 잘못된 `attempt01` manifest는 학습되지 않은 참고 이력으로 보존하고 사용하지 않는다. 기존 hash를 덮어쓰거나 실험 결과를 삭제하지 않았다. 이 변경은 task 구성/학습률/batch/CL 방법 변경이 아니라 공식 학습·평가 간 입력/출력 schema를 일치시키는 수정이다.

공식 README에는 RTX 4090에서 OOM을 피하기 위한 diffusion-model 동결 안내도 있지만, 이를 그대로 적용하면 이번 action-head 학습 분석의 범위가 달라진다. 자동으로 동결하거나 batch를 바꾸지 않으며, 데이터 준비 후 pilot에서 VRAM을 측정하고 필요한 조정은 별도로 결정한다. 기본 환경/CPU 테스트 통과를 전체 GR00T 학습 가능성의 검증으로 해석하지 않는다.
