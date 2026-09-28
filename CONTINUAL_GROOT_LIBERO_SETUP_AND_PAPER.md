# continual-GR00T + LIBERO 설치·데이터·논문 재현 점검

점검일: 2026-09-21
프로젝트 루트: `/root/workspace/continual-GR00T`

## 1. 결론

| 항목 | 상태 | 결론 |
|---|---:|---|
| continual-GR00T 코드 | 완료 | 공식 저장소 `Continual-VLAs/continual-GR00T`, 커밋 `4bfd66ff44a7fbf17dffa86e7c26ae4e0a4565c0` |
| GR00T 실행 환경 | 완료 | Python 3.10.20, PyTorch 2.5.1+cu124, CUDA 12.4 빌드, flash-attn 2.7.1.post4 |
| LIBERO 코드 | 완료 | 사용자가 지정한 `Continual-VLAs/libero`, 커밋 `1af3b848a5b5a1a403b1ff82aa6bbbb937d4fb7d` |
| LIBERO 원본 HDF5 | **완료** | 4개 suite, 총 130/130 파일, 각 파일 50 trajectory, 전부 `libero-v1` |
| LIBERO 시뮬레이터 | 완료 | EGL off-screen 환경 reset 및 64×64 이미지 관측 성공 |
| GR00T용 LeRobot 변환 데이터 | **완료** | `changyeon/libero_cl`의 40/40 task, 원격 5,250개 파일을 완전 다운로드 |
| GR00T 데이터 로더 | **전수 검증 완료** | 40/40 task를 `LeRobotSingleDataset`으로 초기화하고 실제 parquet·영상·transform sample 로드 성공 |
| 모델 checkpoint | 없음 | 이전 요청에 따라 제거됨. 최초 학습 시 `nvidia/GR00T-N1.5-3B`를 받아야 함 |

따라서 “LIBERO 데이터를 온전히 받았는가?”에 대한 정확한 답은 다음과 같다.

- 공식 LIBERO가 배포하는 **원본 HDF5 데이터는 온전히 받았다.**
- continual-GR00T가 학습에 바로 쓰는 **LeRobot 변환 데이터도 온전히 받았고 40개 task 모두 실제 로딩을 통과했다.**

## 2. 저장소에 설치법이 어디까지 나와 있는가

### continual-GR00T

- [`README.md`](README.md#L5): GR00T 설치는 `README_GR00T.md`를 보라고 안내한다.
- [`README.md`](README.md#L7): 데이터·출력 경로를 `bash_scripts/config.sh`에서 설정하는 법을 설명한다.
- [`README.md`](README.md#L23): sequential training/evaluation 명령 생성법과 ER 기본값 1,000을 설명한다.
- [`README_GR00T.md`](README_GR00T.md#L88): Ubuntu, GPU, Python 3.10, CUDA 12.4 요구사항을 설명한다.
- [`README_GR00T.md`](README_GR00T.md#L96): `pip install -e .[base]`와 flash-attn 2.7.1.post4 설치법을 설명한다.
- [`README_GR00T.md`](README_GR00T.md#L121): GR00T 데이터가 LeRobot-compatible schema와 `meta/modality.json`을 가져야 함을 설명한다.

하지만 이 저장소만으로는 다음 과정이 end-to-end로 설명되지 않는다.

- LIBERO 원본 HDF5를 논문에서 사용한 GR00T용 LeRobot 형식으로 변환하는 명령
- 논문 저자가 공개한 `changyeon/libero_cl` 전체를 받는 명령
- continual-GR00T와 LIBERO의 충돌 없는 단일 환경 구성

즉, **GR00T 자체 설치와 continual 명령은 나와 있지만, LIBERO 데이터를 학습 직전 상태까지 만드는 설명은 불완전하다.**

### Continual-VLAs/libero

지정 저장소의 README에는 다음이 있다.

- Python 3.8.13 기반 별도 LIBERO 환경
- `requirements.txt`, PyTorch 1.11.0/cu113, editable install
- `benchmark_scripts/download_libero_datasets.py`를 통한 네 suite 다운로드
- Hugging Face mirror를 쓰는 `--use-huggingface` 옵션

이 안내는 LIBERO 단독 실행 기준이다. GR00T N1.5는 Python 3.10/CUDA 12.4/PyTorch 2.5 계열이므로, README의 오래된 PyTorch 1.11 명령을 그대로 현재 GR00T 환경에 적용하면 안 된다. 현재 환경은 Python 3.10에서 LIBERO 호환 버전을 별도로 고정해 두었다.

공식 링크:

- [continual-GR00T](https://github.com/Continual-VLAs/continual-GR00T)
- [Continual-VLAs/libero](https://github.com/Continual-VLAs/libero)
- [논문 프로젝트 페이지](https://continual-vlas.github.io/forget-me-not/)
- [논문 PDF](https://continual-vlas.github.io/forget-me-not/static/continual_vla.pdf)

## 3. 현재 로컬 환경

활성화:

```bash
cd /root/workspace/continual-GR00T
source activate.sh
```

주요 버전:

| 구성 | 현재 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| NVIDIA driver | 580.173.02 |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| flash-attn | 2.7.1.post4 |
| NumPy | 1.26.4 |
| MuJoCo | 2.3.7 |
| robosuite | 1.4.0 |

경로 설정:

- [`activate.sh`](activate.sh): 가상환경, 프로젝트 경로, LIBERO 설정, EGL 활성화
- [`bash_scripts/config.sh`](bash_scripts/config.sh#L3): raw/LeRobot 데이터와 output의 로컬 경로
- [`gr00t/config.py`](gr00t/config.py#L5): 평가 코드가 읽는 데이터·output 기본 경로와 환경변수 override
- [`.libero/config.yaml`](.libero/config.yaml): BDDL, init state, assets, dataset 경로

## 4. LIBERO 원본 데이터 완전성 검증

위치: `datasets/libero_raw`

| suite | HDF5 수 | 기대 수 | 결과 |
|---|---:|---:|---:|
| `libero_spatial` | 10 | 10 | 통과 |
| `libero_object` | 10 | 10 | 통과 |
| `libero_goal` | 10 | 10 | 통과 |
| `libero_10` | 10 | 10 | 통과 |
| `libero_90` | 90 | 90 | 통과 |
| 합계 | **130** | **130** | **통과** |

추가 무결성 검사 결과:

- 130개 HDF5 모두 열림
- 모든 HDF5의 `data` 아래 trajectory 수가 정확히 50개
- 130개 모두 dataset tag가 `libero-v1`
- 총 용량 약 94 GiB

공식 검사 기준은 [`third_party/LIBERO/scripts/check_dataset_integrity.py`](third_party/LIBERO/scripts/check_dataset_integrity.py#L10)에서 확인할 수 있다. 이 코드는 각 HDF5의 demo 수가 50인지 검사하고, `libero-v1` tag를 확인한다.

다운로드 entry point와 지원 suite는 [`third_party/LIBERO/benchmark_scripts/download_libero_datasets.py`](third_party/LIBERO/benchmark_scripts/download_libero_datasets.py#L10)에서 확인할 수 있다. `libero_100`은 실제로 `libero_90 + libero_10`이다.

참고: 지정 포크의 Hugging Face downloader는 `libero_100/*` 패턴을 요청하지만 mirror의 실제 최상위 경로는 `libero_90/`과 `libero_10/`이다. 이 때문에 `libero_100` 자동 다운로드가 비어 있는 문제가 있어 두 실제 경로를 직접 받아 완전성을 검사했다.

## 5. GR00T용 LeRobot 데이터 상태

위치: `datasets/libero`
출처: `changyeon/libero_cl`

원격 데이터셋은 40개 task, 5,250개 파일, 약 33.2 GiB이다. 인증된 Hugging Face 다운로드가 5분당 API 요청 제한에 걸려 Git LFS로 이어받았으며, 다음을 확인했다.

- task directory: **40/40**
- 원격 tracked file: **5,250/5,250**
- parquet: **1,693개**
- MP4: **3,386개**
- `meta/modality.json`: **40/40**
- 1 KiB 미만의 미수신 LFS pointer parquet/MP4: **0개**
- Git LFS checkout 직후 작업 트리 변경: **0개**
- 데이터 작업 트리 용량: 약 **34 GiB**

원격에는 `stats.json`이 10개만 있었기 때문에 GR00T loader가 나머지 30개 task의 통계를 parquet 전체에서 계산했다. 현재는 **40/40 task에 `meta/stats.json`이 존재**한다. 이 로컬 생성 파일까지 합친 현재 파일 수는 5,280개다.

40개 task 각각에 대해 실제로 다음을 수행했고 실패는 0개였다.

- `LeRobotSingleDataset` 초기화 및 trajectory index 구성
- 전체 parquet를 통한 metadata/statistics 로드
- 첫 sample의 front/wrist MP4 frame decode
- language annotation과 LIBERO transform 적용
- state tensor `(1, 64)` 확인
- action tensor `(16, 32)` 확인

따라서 데이터는 파일 수뿐 아니라 **GR00T 학습 loader 기준으로도 전수 검증 완료** 상태다. Hugging Face token은 로컬 credential에만 있으며 채팅이나 이 문서에는 기록하지 않았다.

## 6. 논문의 continual learning 실험 설정

논문: *Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning* (2026-03-04 preprint)

### 실험 단위

- LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-10을 사용한다.
- suite마다 별도 모델 하나를 만든다. 즉 네 suite를 하나의 40-task stream으로 합치지 않는다.
- 각 suite 안의 10개 task를 정해진 순서로 하나씩 학습한다.
- task `k` 학습은 task `k-1`의 최종 checkpoint에서 시작한다.
- 모든 비교 모델은 같은 task order를 사용한다.
- 학습 데이터는 Kim et al. (2024)의 filtered LIBERO dataset이다. 이 저장소에서는 GR00T용 공개본 `changyeon/libero_cl`을 사용하도록 경로와 task 이름이 구성돼 있다.

순차 checkpoint 연결 구현은 [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L652)에서 확인할 수 있다. task loop를 돌며 이전 task의 `checkpoint-{max_steps}`를 다음 task의 base model로 지정한다.

논문 task order를 코드에서 확인할 위치:

- LIBERO-10: [`bash_scripts/generate_train_commands.sh`](bash_scripts/generate_train_commands.sh#L20)
- LIBERO-Object: [`bash_scripts/generate_train_commands.sh`](bash_scripts/generate_train_commands.sh#L33)
- LIBERO-Spatial: [`bash_scripts/generate_train_commands.sh`](bash_scripts/generate_train_commands.sh#L46)
- LIBERO-Goal: [`bash_scripts/generate_train_commands.sh`](bash_scripts/generate_train_commands.sh#L59)

### GR00T 모델과 학습 대상

| 구성 | 논문 설정 |
|---|---|
| 모델 | GR00T N1.5, 약 3B parameters |
| vision encoder | pretrained SigLIP, frozen |
| language model | pretrained Qwen3-1.7B, frozen |
| action head | pretrained Flow Matching DiT, full fine-tuning |
| fine-tuning 전략 | vision/LLM 고정, action projector와 diffusion model 학습 |

코드 대응:

- 기본 base model: [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L70)
- `tune_llm=False`, `tune_visual=False`: 같은 파일 [74행](scripts/gr00t_sequential_finetune.py#L74)
- `tune_projector=True`, `tune_diffusion_model=True`: 같은 파일 [80행](scripts/gr00t_sequential_finetune.py#L80)
- 실제 모델 생성과 tune flag 전달: 같은 파일 [336행](scripts/gr00t_sequential_finetune.py#L336)

`README_GR00T.md`의 RTX 4090 일반 권고에는 OOM 방지를 위해 diffusion model을 끄라는 문구가 있지만, 그것은 일반 fine-tuning 예시다. **논문 재현 설정은 action head를 full fine-tuning하므로 diffusion model을 켜는 현재 continual script 기본값이 논문과 맞다.** 4090에서 OOM이 발생하면 논문 설정을 바꾸기 전에 batch/memory 최적화부터 적용해야 한다.

### 학습 하이퍼파라미터

| 값 | 논문 | 저장소 구현 |
|---|---:|---:|
| learning rate | `1e-4` | `1e-4` |
| batch size | `16` | command generator `16` |
| budget/task | `10,000` gradient steps | 기본 `10,000` |
| optimizer | AdamW | `adamw_torch` |
| weight decay | `1e-5` | `1e-5` |
| LR scheduler | cosine decay | `cosine` |
| warmup | 논문 표에 미기재 | 코드 `0.05` |
| precision | 논문 표에 미기재 | bf16 + tf32 |
| Adam betas | 논문 표에 미기재 | `(0.95, 0.999)` |
| seed | 결과는 여러 seed 평균 | generator 기본 `42` |

정의 위치:

- 기본 학습값: [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L57)
- optimizer/scheduler/precision: [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L392)
- 실제 논문형 command의 batch, step, ER memory: [`bash_scripts/generate_train_commands.sh`](bash_scripts/generate_train_commands.sh#L6), [206행](bash_scripts/generate_train_commands.sh#L206)

### Experience Replay(ER)

논문의 기본 ER:

- 지난 각 task에서 무작위 transition 1,000개를 고정 저장한다.
- 같은 저장 샘플을 이후 task들에서 계속 재사용한다.
- 현재 task와 **모든 과거 task를 합친 replay buffer**를 1:1로 sampling한다.
- 1,000개/task는 task 전체의 약 15–20%이며, 논문 표에서는 약 20%로 표기한다.
- buffer ablation은 10, 100, 1,000 transition/task, 대략 0.2%, 2%, 20%이다.

코드 흐름:

1. [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L181)의 `saved_indices`가 task별 표본 index를 유지한다.
2. 같은 파일 [260행](scripts/gr00t_sequential_finetune.py#L260)에서 이전 task 각각을 `n_memories`만큼 deterministic random subset으로 만든다.
3. [296행](scripts/gr00t_sequential_finetune.py#L296)에서 과거 subset들을 하나의 replay dataset으로 합친다.
4. [159행](scripts/gr00t_sequential_finetune.py#L159)의 collator가 current samples와 replay samples를 합친다.
5. [429행](scripts/gr00t_sequential_finetune.py#L429)에서 current batch와 같은 크기의 replay loader를 만들어 1:1 비율을 구현한다.

### LIBERO 입력·출력 표현

학습 configuration은 [`gr00t/experiment/data_config.py`](gr00t/experiment/data_config.py#L1159)에 있다.

- 입력 영상: front view + left wrist view
- state: absolute EEF position, absolute EEF rotation, gripper
- action: delta EEF position, delta EEF rotation, gripper
- observation horizon: 1
- action horizon: 16
- 영상: 0.95 crop → 224×224 resize → color jitter
- state/action: min-max normalization
- model padded maximum: state 64, action 32

훈련 command는 `--data-config libero --embodiment-tag libero`를 사용한다: [`bash_scripts/generate_train_commands.sh`](bash_scripts/generate_train_commands.sh#L206).

### 평가

- checkpoint `k`를 task `1..k`에 평가해 lower-triangular success matrix를 만든다.
- 논문 지표는 평균 success rate(SR)와 Negative Backward Transfer(NBT)다.
- NBT가 양수면 forgetting이며 낮을수록 좋다. 음수는 과거 task 성능이 오히려 개선됐다는 뜻이다.
- simulator 기본 평가는 task당 50 rollout이다.
- 평가 code는 suite별 최대 horizon을 Spatial 220, Object 280, Goal 300, LIBERO-10 520으로 둔다.
- action horizon 16 중 5 step만 실행하고 다시 계획한다.

코드 위치:

- checkpoint와 학습 완료 task까지의 평가 명령 생성: [`bash_scripts/generate_eval_commands.sh`](bash_scripts/generate_eval_commands.sh#L39)
- 50 rollout, horizon, seed: [`scripts/eval_policy_libero.py`](scripts/eval_policy_libero.py#L27)
- task 초기 상태와 환경 생성: [`scripts/eval_policy_libero.py`](scripts/eval_policy_libero.py#L198)
- 관측 구성과 action query: [`scripts/eval_policy_libero.py`](scripts/eval_policy_libero.py#L293)
- success rate 저장: [`scripts/eval_policy_libero.py`](scripts/eval_policy_libero.py#L347)

### 논문 기본 결과(M=1,000)

| suite | GR00T SR | GR00T NBT |
|---|---:|---:|
| LIBERO-Spatial | 0.940 ± 0.013 | 0.007 ± 0.009 |
| LIBERO-Object | 0.975 ± 0.004 | 0.019 ± 0.013 |
| LIBERO-Goal | 0.943 ± 0.004 | 0.023 ± 0.017 |
| LIBERO-10 | 0.820 ± 0.017 | 0.059 ± 0.035 |
| 4-suite 평균 | **0.919 ± 0.011** | **0.027 ± 0.021** |

## 7. 재현 전에 반드시 확인할 코드상 주의점

### 7.1 평가 data config 불일치(로컬 수정 완료)

원본 저장소의 훈련 generator는 `--data-config libero`를 쓰지만, 평가 generator는 `--args.data-config libero90`을 썼다. 두 config의 key schema가 다르다.

- `libero`: `video.front_view`, `video.left_wrist_view`, EEF state/action
- `libero90`: `video.image`, `video.wrist_image`, scalar x/y/z/roll/pitch/yaw schema

현재 받은 `changyeon/libero_cl`과 `eval_policy_libero.py`의 observation key는 `libero` 쪽이다. 따라서 로컬 [`bash_scripts/generate_eval_commands.sh`](bash_scripts/generate_eval_commands.sh#L58)을 `libero`로 수정했다. 두 schema 정의는 [`gr00t/experiment/data_config.py`](gr00t/experiment/data_config.py#L1159), [1259행](gr00t/experiment/data_config.py#L1259)에서 비교할 수 있다.

### 7.2 논문 batch size 16과 구현의 effective batch

generator와 `TrainingArguments`의 batch size는 16이다. 그러나 ER collator는 current 16개에 replay 16개를 붙이므로 모델에 전달되는 tensor의 첫 차원은 32가 된다. 논문의 “batch size 16”이 total 16을 의미한다면 구현은 effective batch가 두 배다. 논문 공개 코드 그대로 재현할지, total batch 16(8+8)으로 해석할지는 첫 본 실험 전에 작은 dry run에서 loss tensor와 GPU memory를 확인하고 기록해야 한다.

### 7.3 4090 메모리

논문 설정은 action head full fine-tuning이다. upstream GR00T README는 4090에서 diffusion model fine-tuning 시 OOM 가능성을 경고한다. 현재 GPU는 24 GB 4090 한 장이므로 batch 16(+ ER 결합 시 실제 32)이 그대로 들어갈 가능성은 낮다. gradient checkpointing, accumulation, 또는 source별 microbatch를 조정하면 논문과 완전히 같은 optimizer-step 의미가 달라질 수 있으므로 변경값을 실험 로그에 남겨야 한다.

### 7.4 `no_resampling` 분기

논문 기본 1:1 ER은 `no_resampling=False` 분기다. `--no-resampling`을 켜면 current와 과거 task 모두를 mixture dataset으로 만들며 논문 기본 ER loader와 다른 학습 분포가 된다.

## 8. 권장 실행 순서

1. ~~Hugging Face 로그인 후 `datasets/libero`를 40/40 task까지 다운로드한다.~~ 완료
2. ~~모든 task의 metadata, parquet, 양쪽 camera MP4를 원격 manifest와 대조하고 loader로 전수 검증한다.~~ 완료
3. ~~평가 generator가 `--args.data-config libero`를 출력하도록 맞춘다.~~ 로컬 수정 완료
4. 한 task, 수 step dry run으로 실제 batch shape, GPU peak memory, checkpoint 저장을 검증한다.
5. 한 checkpoint로 LIBERO rollout 1회 실행해 관측 key/action scale을 확인한다.
6. 그 뒤 suite 하나당 10 task × 10,000 step 본 실험을 시작한다.
7. 각 task checkpoint를 지금까지 학습한 모든 task에 50회씩 평가해 SR matrix와 NBT를 계산한다.

논문형 training command 생성 예:

```bash
cd /root/workspace/continual-GR00T
source activate.sh
./bash_scripts/generate_train_commands.sh libero_spatial 10000 ER 1000 10000 42 False False s42
```

이 명령은 실행하지 않고 Python command를 출력한다. 출력값을 검토한 후 실행해야 한다.

평가 command 생성 예:

```bash
./bash_scripts/generate_eval_commands.sh libero_spatial 10000 ER 1000 10000 42 False False s42 9 10000
```

로컬 평가 generator는 위에서 설명한 schema 문제를 수정한 상태다.

## 9. 핵심 코드 지도

| 확인할 내용 | 파일 위치 |
|---|---|
| 저장소 사용법·명령 인자 | [`README.md`](README.md) |
| GR00T 설치·환경·일반 fine-tuning | [`README_GR00T.md`](README_GR00T.md) |
| 로컬 데이터/output 경로 | [`bash_scripts/config.sh`](bash_scripts/config.sh), [`gr00t/config.py`](gr00t/config.py) |
| suite별 논문 task order | [`bash_scripts/generate_train_commands.sh`](bash_scripts/generate_train_commands.sh) |
| sequential checkpoint 전달 | [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L652) |
| ER buffer 추출·고정 index | [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L181) |
| current:replay 1:1 결합 | [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L159) |
| optimizer·scheduler·정밀도 | [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L392) |
| 모델 freeze/tune flag 전달 | [`scripts/gr00t_sequential_finetune.py`](scripts/gr00t_sequential_finetune.py#L336) |
| LIBERO modality·transform | [`gr00t/experiment/data_config.py`](gr00t/experiment/data_config.py#L1159) |
| LIBERO simulator 평가 | [`scripts/eval_policy_libero.py`](scripts/eval_policy_libero.py) |
| Foxglove 실시간/MCAP rollout 시각화 | [`FOXGLOVE_ROLLOUT.md`](FOXGLOVE_ROLLOUT.md) |
| Foxglove telemetry 구현 | [`gr00t/eval/foxglove_rollout.py`](gr00t/eval/foxglove_rollout.py) |
| checkpoint별 평가 명령 | [`bash_scripts/generate_eval_commands.sh`](bash_scripts/generate_eval_commands.sh) |
| 공식 HDF5 무결성 기준 | [`third_party/LIBERO/scripts/check_dataset_integrity.py`](third_party/LIBERO/scripts/check_dataset_integrity.py) |
| LIBERO 데이터 downloader | [`third_party/LIBERO/benchmark_scripts/download_libero_datasets.py`](third_party/LIBERO/benchmark_scripts/download_libero_datasets.py) |
