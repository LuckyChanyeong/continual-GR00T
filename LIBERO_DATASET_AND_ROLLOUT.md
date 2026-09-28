# LIBERO-Spatial 데이터와 rollout 구성

작성: 2026-09-27. 대상: 현재 `none/step10000_bs16_adamw_bnb_8bits42` continual run과 Task 0–5 분석.

## 1. 벤치마크 구조

LIBERO는 언어 지시가 주어진 continual robot manipulation benchmark다. 총 130개 task는 `LIBERO-Spatial` 10개, `LIBERO-Object` 10개, `LIBERO-Goal` 10개와 `LIBERO-100`으로 구성되고, LIBERO-100은 사전학습용 `LIBERO-90`과 downstream lifelong 평가용 `LIBERO-10`으로 나뉜다. 현재 실험은 `LIBERO-Spatial`만 사용한다. 이 suite의 10개 task는 같은 Panda 기반 tabletop 조작에서 물체의 시작 위치 관계를 바꾼다. 각 task의 BDDL 파일이 scene fixture, object, 초기 관계와 성공 goal을 정의한다.

Task 0–5는 모두 검은 bowl을 plate에 놓지만 시작 위치가 다르다: plate와 ramekin 사이, ramekin 옆, table 중앙, cookie box 위, wooden cabinet top drawer 안, ramekin 위.

## 2. 실제 continual 학습 데이터 수

공식 raw HDF5는 각 task에 50개 demonstration이 있다. 현재 checkpoint가 실제 읽은 `datasets/libero/<task>` LeRobot 데이터는 별도로 내려받은 변환본이며 episode 수가 더 적다.

| Task | 시작 위치 | Raw demo | 학습 demo | 학습 frame/sample | 길이 min/mean/max | 10,000 step에서 관측된 epoch |
|---:|---|---:|---:|---:|---:|---:|
| 0 | plate와 ramekin 사이 | 50 | 45 | 4,487 | 75 / 99.71 / 168 | 35.59 |
| 1 | ramekin 옆 | 50 | 45 | 5,940 | 109 / 132.00 / 189 | 26.88 |
| 2 | table 중앙 | 50 | 46 | 5,343 | 93 / 116.15 / 143 | 29.94 |
| 3 | cookie box 위 | 50 | 43 | 4,287 | 86 / 99.70 / 129 | 37.31 |
| 4 | wooden cabinet top drawer 안 | 50 | 42 | 6,257 | 127 / 148.98 / 182 | 25.51 |
| 5 | ramekin 위 | 50 | 39 | 4,472 | 86 / 114.67 / 154 | 35.71 |
| 합계 |  | 300 | 260 | 30,786 |  |  |

로컬 문서에는 LeRobot 변환본의 출처가 `changyeon/libero_cl`로 기록되어 있다. 현재 저장소에는 raw 50개 중 어떤 기준으로 episode가 제외되었는지 provenance가 없다. 제공된 변환 스크립트 자체는 성공 여부를 필터링하지 않는다. 따라서 감소분을 quality filtering이라고 단정할 수 없다.

학습 시 `num_demos` 제한을 주지 않으므로 위 변환본의 모든 episode를 사용한다. `use_rl=False` 경로에서 각 frame이 하나의 base sample이 된다. 각 sample은 256×256 front/wrist RGB, 현재 end-effector와 gripper state, 언어 지시를 입력으로 사용하며, 현재부터 16 step의 7차원 action chunk를 target으로 구성한다. episode 끝을 넘어가는 relative position/rotation action은 zero padding되고, absolute gripper action은 마지막 값으로 padding된다.

`cl_type=none`에서는 Task k 단계가 현재 task 데이터만 10,000 optimizer step 학습하고, Task k-1 checkpoint를 초기 모델로 불러온다. Batch size는 16이다. 따라서 위 frame은 한 번만 쓰이는 것이 아니라 표의 epoch 수만큼 반복 노출된다.

Gradient-first 분석의 task당 20 demonstration은 위 학습 episode 중 고정 선택한 부분집합이다. 각 episode에서 4 frame을 사용하므로 task당 80 probe다. 별도 held-out set이 아니다.

## 3. Rollout 환경 생성과 한 episode의 진행

1. LIBERO benchmark에서 task를 불러오고 해당 BDDL로 scene, object, 초기 관계, language instruction과 성공 goal을 정한다.
2. 256×256 off-screen environment를 만들고 현재 코드 기본값인 NumPy/environment seed 7을 설정한다.
3. task마다 제공되는 `pruned_init` bank를 읽는다. Task 0–5 각각 shape은 `(50, 92)`이고 50개 상태는 모두 서로 다르다.
4. 현재 결과는 task당 30 episode다. Episode `i`에서 `env.reset()` 후 `initial_states[i]`를 주입하므로 50개 중 앞의 30개만 사용한다.
5. 물체 안정화를 위해 첫 10 simulator step은 zero motion과 open-gripper dummy action을 실행한다.
6. 정책 입력은 front image, wrist image, end-effector position/quaternion, gripper state와 같은 language instruction이다. 이미지는 학습 방향과 맞추기 위해 180도 회전한다.
7. GR00T는 4번의 flow denoising으로 16-step action chunk를 만들며, 앞의 5 step만 실행한 뒤 다시 관측하고 재계획한다.
8. LIBERO-Spatial은 settling 이후 최대 220 policy step 안에 BDDL goal이 만족되면 성공하고 즉시 종료한다. 아니면 실패다.

## 4. 무엇이 같고 무엇이 다른가

### 같은 task의 episode 사이에서 고정

- BDDL scene, object 종류, target object와 성공 조건
- language instruction
- Panda robot, controller와 physics 설정
- camera 종류와 256×256 해상도
- 10-step settling, 220 policy-step 제한, 5-step replan, 4-step denoising
- 같은 checkpoint를 평가하는 동안의 model parameter

### 같은 task의 episode마다 변화

- `initial_states[episode_idx]`: robot joint state와 object pose 등을 담은 92차원 simulator state
- 그 상태에서 렌더링된 영상과 robot state
- 정책이 생성한 action, 이후 trajectory, 종료 시점과 성공 여부
- flow policy의 초기 action noise

### 같은 task를 서로 다른 checkpoint에서 비교할 때

- task 정의와 episode index 0–29의 initial-state bank는 같다.
- 달라져야 하는 주 변수는 model checkpoint다.
- 그러나 현재 action head는 매 action chunk를 `torch.randn` noise에서 시작하고, 평가 스크립트는 NumPy/environment seed만 설정하며 PyTorch seed는 명시적으로 설정하지 않는다. 따라서 checkpoint 간 episode는 같은 초기 simulator state를 사용하지만 flow-noise까지 같은 완전한 paired rollout은 아니다.
- 로그에는 실제 사용 seed가 별도 필드로 저장되지 않았다. 현재 코드와 command generator 기준 기본값은 7이지만, 기존 결과의 exact seed provenance는 완전하지 않다.

## 5. 해석상 주의점

현재 30-rollout 성공률은 같은 initial-state index를 사용하므로 대체로 공정한 checkpoint 비교지만, stochastic action noise가 대응되지 않아 작은 차이는 평가 잡음일 수 있다. 엄밀한 후속 평가에서는 `(task, episode, replan)`별 PyTorch noise seed를 고정하고, 각 checkpoint에 정확히 같은 30개 초기 상태와 noise를 적용해야 한다. 여러 rollout seed를 반복한 뒤 episode-paired 차이와 신뢰구간을 보고하는 것이 적절하다.

## 6. 핵심 용어

### BDDL

BDDL은 LIBERO task를 기술하는 PDDL 계열의 텍스트 문제 정의다. 데이터 trajectory 자체가 아니라 simulator가 어떤 scene을 만들고 무엇을 성공으로 볼지 지정한다. 파일에는 language instruction, 배치 region과 좌표 범위, fixture와 movable object, 관심 object, symbolic initial predicates, goal predicates가 들어 있다.

LIBERO의 multi-stage/long-horizon demonstration에는 frame별 `subtask_id`, subtask language, 또는 subtask 경계가 제공되지 않는다. Raw HDF5는 episode 전체의 language instruction과 observation/action/reward/done/state를 저장하고, 현재 LeRobot 변환본도 동일한 전체 instruction의 `task_index`를 모든 frame에 반복한다. BDDL의 여러 goal predicate는 후보 subgoal을 정의하는 데 사용할 수 있지만, trajectory에서 각 predicate를 언제 달성했는지나 실행 순서를 라벨링하지 않는다. Subtask 분석에는 simulator state를 replay하며 predicate를 매 frame 평가하거나, 별도의 수동/휴리스틱 segmentation이 필요하다.

Task 0의 핵심은 `Init: On(akita_black_bowl_1, main_table_between_plate_ramekin_region)`과 `Goal: And(On(akita_black_bowl_1, plate_1))`이다. BDDL은 bowl이 어느 관계에 있어야 하는지를 정하고, initial-state bank는 그 관계를 만족하는 정확한 수치 상태를 제공한다.

### Initial-state bank

각 task의 `.pruned_init` 파일에 저장된 평가 시작 상태 모음이다. Task 0–5에서는 각각 50개의 서로 다른 92차원 flattened MuJoCo simulator state다. 이 벡터는 robot과 object의 실제 simulation configuration을 복원하며, 이미지나 expert action trajectory가 아니다. Episode `i`는 `env.reset()` 뒤 `set_init_state(initial_states[i])`로 해당 상태를 불러온다. 현재 30-rollout 평가는 index 0–29만 사용한다.

### Dummy action

현재 값은 `[0, 0, 0, 0, 0, 0, -1]`이다. 앞의 3개는 end-effector translation, 다음 3개는 rotation 명령이며 모두 0이다. 마지막 `-1`은 Panda gripper open 명령이다. Initial state 주입 직후 10 control step 동안 실행해 중력과 contact dynamics로 물체가 안정되도록 한다. 제어 주파수 20 Hz 기준 약 0.5초다. 따라서 정책이 처음 보는 상태는 저장된 initial state 그 자체가 아니라 이 settling을 거친 상태다.

### 성공 조건

Task 0–5의 BDDL goal은 모두 `And(On(akita_black_bowl_1, plate_1))`이다. 매 environment step마다 goal predicate를 평가한다. 현재 `On(bowl, plate)` 구현은 bowl과 plate가 contact하고, bowl 중심이 plate보다 높으며, 두 물체의 XY 중심 거리가 0.03 m 미만일 때 참이다. 참이 되는 순간 `done=True`로 rollout을 성공 처리한다. Gripper를 놓았는지, expert action과 비슷한 경로를 사용했는지, 동작이 매끄러운지는 성공 조건에 포함되지 않는다. 제한 시간까지 predicate가 한 번도 참이 아니면 실패다.

## 7. Step과 epoch 용어 구분

- Training sample/frame: demonstration의 한 시점. 현재 관측과 이후 16-step expert action chunk로 구성한다.
- Training optimizer step: batch 하나로 forward/backward 후 optimizer가 parameter를 한 번 갱신하는 단위다. 현재 batch size 16, gradient accumulation 1이므로 보통 16 samples마다 1 optimizer step이다.
- Epoch: 해당 task의 모든 training samples를 평균적으로 한 번 순회한 양이다. Task 0은 4,487 samples이므로 약 `ceil(4487/16)=281` optimizer steps가 1 epoch이고, 10,000 optimizer steps는 약 35.59 epochs다.
- Rollout environment/control step: simulator에 7차원 action 하나를 적용하고 다음 관측으로 진행하는 단위다. 현재 control frequency 20 Hz이므로 한 step은 약 0.05초다. 10 settling steps 뒤 최대 220 policy steps를 실행하므로 정책 제어 시간은 약 11초다.
- Replan: 정책은 16-step action chunk를 예측하지만 5 environment steps만 실행한 뒤 새 관측으로 다시 예측한다. 최대 220 steps이면 최대 약 44번 action chunk를 요청한다.
- Flow denoising step: action chunk 하나를 생성하는 모델 내부 수치 적분 단계다. 현재 4회이며 environment step이나 optimizer step과 별개다.

`220 step까지 실패`라는 표현은 부정확하다. 정확한 의미는 settling 이후 최대 220개의 policy action을 simulator에 적용하는 동안 BDDL 성공 predicate가 한 번이라도 참이면 즉시 성공하고, 끝까지 참이 아니면 timeout failure로 기록한다는 것이다. Rollout에는 epoch 개념이 없다.

## 8. Batch size 16의 실제 구성

`cl_type=none`의 Task k 학습에서 한 batch는 현재 Task k 데이터의 서로 다른 base frame sample 최대 16개다. 16개 demonstration을 뜻하지 않는다. 한 sample은 한 episode의 시점 `t`를 기준으로 front/wrist RGB 한 장씩, 현재 robot state, 동일 task language instruction, expert action `t:t+16`, action/state mask와 embodiment 정보를 포함한다. 원본 차원은 state 8, action `16×7`이며 transform 뒤 GR00T의 최대 차원으로 padding되고 mask가 실제 차원을 표시한다.

Dataset index는 모든 episode의 모든 frame을 펼친 목록이다. Training sampler가 이를 epoch마다 섞으므로 한 batch에는 서로 다른 episode의 frame이 섞일 수도 있고 같은 episode의 서로 다른 frame이 함께 들어갈 수도 있다. 연속 frame은 별도 sample이므로 전체 frame 수는 독립 trajectory 수가 아니다. Task 0은 45 trajectories에서 나온 4,487 frame samples다.

한 optimizer step에서는 16 samples 각각에 image augmentation과 flow noise/time을 적용하고 masked flow-matching error를 모든 유효 action element에 대해 합산·정규화해 하나의 scalar loss를 만든 뒤 backward한다. Epoch 마지막 batch는 나머지 sample만 포함할 수 있다. Task 0은 `4487 = 16×280 + 7`이므로 drop-last가 꺼진 현재 설정에서는 280개 full batch와 7-sample 마지막 batch, 총 281 optimizer steps가 약 1 epoch다.

## 9. 하나의 관측에 가능한 action chunk의 비유일성

`action horizon=16`은 한 target chunk 안에 시간순 action이 16개 있다는 뜻이다. 가능한 target chunk가 16개라는 뜻이 아니다. Dataset은 sample `(episode, t)`마다 해당 expert가 실제 실행한 `action[t:t+16]` 한 chunk만 제공한다. 그러나 동일하거나 유사한 상태에서 bowl에 접근하는 방향, 속도, grasp pose와 경로가 다른 여러 action chunk가 모두 task 성공으로 이어질 수 있으므로 행동 수준의 정답은 본질적으로 유일하지 않다.

Flow matching은 조건부 action distribution을 표현할 수 있지만, 동일 관측에 대한 다양한 expert 행동이 데이터에 충분히 들어 있어야 그 다봉성을 학습할 수 있다. 현재 각 정확한 관측에는 사실상 하나의 recorded expert chunk만 있으므로 demonstration flow loss는 task 성공 가능성 자체보다 기록된 expert 행동과의 적합도를 측정한다. 따라서 valid한 다른 action을 생성한 정책도 demo loss가 높을 수 있고, demo loss가 낮아도 closed-loop rollout은 실패할 수 있다. Gradient conflict와 forgetting을 해석할 때 flow loss는 보조 지표이며 rollout success를 대체하지 않는다.
