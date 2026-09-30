# Continual VLA 실험 환경 사용법

기준 계획은 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), 실행 설정은 [configs/continual_analysis.json](configs/continual_analysis.json)이다. 데이터·모델 가중치는 이번 환경 구성에서 다운로드하지 않았으며 실제 학습도 실행하지 않았다.

## 1. 환경 활성화

```bash
cd /home/chanyeong/cl_vla_2026/chanyeong/workspace/continual-GR00T
source .envrc
python scripts/continual_experiment.py doctor
```

프로젝트 안의 Miniconda 환경 `.conda/envs/gr00t` 하나를 사용한다. venv를 중첩하지 않는다. `.envrc`는 GPU 0, EGL device 0, 프로젝트 내부 캐시/설정 경로를 지정한다. W&B는 기본 offline이며 기존 학습 하이퍼파라미터는 변경하지 않는다.

주요 버전: Python 3.10.21, PyTorch 2.5.1/CUDA 12.4, Transformers 4.51.3, flash-attn 2.7.1.post4, robosuite 1.4.0, MuJoCo 2.3.7, scikit-learn 1.7.2. 핵심 학습 패키지를 업그레이드하지 않고 시뮬레이션·분석 의존성을 추가했다.

LIBERO는 `third_party/LIBERO`에 설치했다. 기준 commit은 `8f1084e3132a39270c3a13ebe37270a43ece2a01`이다. 공식 저장소에 포함된 시뮬레이터 assets/초기 상태는 환경의 일부이며, 학습 demonstration dataset과 다르다.

## 2. 데이터 준비 전 가능한 작업

```bash
# 모델/데이터를 로딩하지 않고 36개 실행 manifest 생성
python scripts/continual_experiment.py prepare --attempt 02

# CPU 단위 테스트
python -m pytest -q tests/test_continual_experiment.py

# GPU 0의 작은 텐서 연산 + 64×64 EGL 렌더링 + dummy action 1 step
# 학습/정책 추론/영상 저장/데이터 다운로드가 아님
python scripts/smoke_libero_environment.py
```

기본 조합은 3 suites × (canonical + order seed 0/1/2) × 학습 seed 42/43/44 = 36 runs이다. 모든 run은 canonical task ID 0~5만 학습한다. `task_order.json`을 직접 수정하지 말고 설정을 바꾼 뒤 새 attempt로 생성한다.

```bash
python scripts/continual_experiment.py prepare --config configs/continual_analysis.json --attempt 03
```

## 3. 나중에 넣을 데이터

기본 위치는 `data/libero/<task_name>/`이다. raw LIBERO HDF5만 놓는 것이 아니라 **이 저장소의 LeRobot 형식**이어야 한다.

```text
data/libero/<task_name>/
  meta/info.json
  meta/modality.json
  meta/episodes.jsonl
  meta/tasks.jsonl
  meta/stats.json
  data/...
  videos/...          # 해당 dataset의 info/modality에 지정된 영상
```

다른 디스크에 데이터를 둘 경우 JSON 설정의 `dataset_root`를 절대 경로로 바꾸고 새 manifest를 만든다. `doctor`는 아직 없는 데이터 위치를 보고할 뿐 다운로드하지 않는다. `run --execute`는 필요한 metadata와 data 디렉터리가 없으면 모델 로딩 전에 중단한다. metadata 검사 통과가 실제 parquet/video 내용까지 정상임을 보장하지는 않으며 데이터 준비 후 pilot이 필요하다.

특히 object의 공식 ID 0~5는 alphabet_soup, cream_cheese, salad_dressing, bbq_sauce, ketchup, tomato_sauce 순이다. 기존 bash 스크립트의 알파벳순 목록과 다르므로 이번 실험은 아래의 manifest 진입점을 사용한다. 공식 LIBERO task catalog와 이름을 매번 대조한다.

## 4. 실행 명령 미리보기와 실제 실행

```bash
# 기본은 dry-run: 명령만 출력
python scripts/continual_experiment.py run \
  --manifest output/continual_analysis/libero_spatial__ER__order00__oseedna__seed42__attempt02/task_order.json

# 전체 조합 명령 확인; GPU 병렬 실행을 하지 않음
python scripts/continual_experiment.py run \
  --index output/continual_analysis/experiment_index__attempt02.json
```

데이터와 초기 모델이 준비되고 pilot을 통과한 뒤에만 같은 명령에 `--execute`를 추가한다. index 실행은 GPU 0에서 run을 순차적으로 실행하며, 오류가 나면 다음 run으로 넘어가지 않는다.

각 phase는 이전 phase의 최종 checkpoint 가중치를 불러오고 AdamW/scheduler는 새로 만든다. 학습은 기본 10,000 step, ER, current batch 16(+이전 task replay 16), memory 1000을 유지한다. phase 5 후 종료한다. EMA/best 선택은 추가하지 않았다.

학습 경로 전체의 자동 재개는 아직 제공하지 않는다. 이미 만들어진 `task_i`를 발견하면 덮어쓰지 않고 중단한다. 실패한 학습을 다시 시작할 때는 원인을 해결하고 새 attempt를 사용한다. 원본 Trainer의 checkpoint 내부 resume과 task chain resume을 혼용하지 않는다.

## 5. 진단 및 저장 비용

- phase-local 500·1000·…·10000에서 실제 적용된 AdamW moment 기반 방향 `train_actual`을 FP32로 저장한다. LR과 decoupled weight decay를 합친 실제 파라미터 차이는 기본 저장 대상이 아니다.
- 같은 시점에서 이전/현재 task의 대응 진단 batch로 `task_diagnostic`을 계산한다. 동일한 현재 optimizer state의 복사 계산만 사용하며 실제 optimizer step을 추가하지 않는다.
- 첫 phase는 현재 task만, 이후 phase는 직전 task와 현재 task를 비교한다. 기본 진단 batch 수는 1이고 샘플 수는 기존 학습 batch 크기이다. sample indices와 noise/augmentation용 seed를 저장한다. 단일 batch 결과를 task 전체의 기울기라고 단정하지 않는다.
- conflict는 같은 모델/optimizer 상태에서 두 task 방향을 비교한다. drift는 이전 phase 최종 시점에서의 같은 task 방향과 다음 phase 각 기록 시점을 비교한다. 전체/모듈별 cosine, relative drift, norm을 원 좌표에서 계산한다.
- 진단 동안 RNG, gradients, train/eval flag, buffers를 보관·복원한다. Fisher도 동일한 격리 호출을 사용하지만 기존 `compute_fisher` 함수 본문/전체 dataset/batch 8/shuffle/제곱 평균/dtype은 바꾸지 않는다.
- Fisher는 현재 task의 전체 dataset을 phase 종료 시 한 번 계산한다. ER이라고 Fisher를 replay 혼합 batch로 바꾸지 않는다. EWC 방식으로 학습하는 경우와 Fisher를 분석용으로만 수집하는 경우를 구분한다.
- 전체 파라미터 벡터를 저장하므로 비용이 크다. `parameter_manifest.json`에 실제 차원, `storage_estimate__phaseXX.json`에 남은 gradient/Fisher 저장량 하한을 기록한다. 하한에도 디스크 공간이 부족하면 학습 전에 중단하고, 저장 중에도 여유 공간을 검사한다. 모델/optimizer checkpoint와 다른 run의 공간은 이 하한에 포함되지 않는다.
- 기본 6-phase run은 120개의 actual + 220개의 task diagnostic 벡터와 6개의 Fisher map을 저장한다. 차원을 P라고 할 때 FP32 기준 대략 `346 × 4P bytes`에 checkpoint 공간이 추가된다. 첫 데이터 pilot에서 차원·VRAM·저장량을 확인하고 큰 output 디스크 또는 별도 합의한 압축 설계를 정한다. 저장 공간이 부족하다고 몰래 차원을 줄이지 않는다.

## 6. 평가와 재개

학습 manifest 실행은 phase마다 최종 checkpoint로 지금까지 본 task를 각각 30 rollout 평가한다. 영상과 debug PNG는 기본 생성하지 않는다.

```bash
# phase 2의 평가 명령만 확인
python scripts/continual_experiment.py evaluate \
  --manifest output/continual_analysis/libero_spatial__ER__order00__oseedna__seed42__attempt02/task_order.json \
  --phase 2

# 실제 재평가는 위 명령에 --execute 추가
```

평가 결과는 episode별 JSONL과 요약 JSON으로 저장한다. 같은 checkpoint·평가 설정으로 재실행하면 완료한 episode는 건너뛰고 미완료/error episode만 다시 실행한다. 30개가 모두 끝나야 완료 결과로 집계한다. 오류를 성공률 0인 rollout로 치환하지 않는다.

평가 base seed는 기존 7이다. 중단 재개와 실행 순서의 영향을 제거하기 위해 policy/noise RNG는 `7 + canonical_task_id × 1000003 + episode_id`로 대응시킨다. 환경 seed 7, 초기 상태 ID 0~29, inference/timeouts/전처리는 유지한다. 이는 기존 평가의 단일 연속 RNG stream과는 다르며, 모든 비교 조건에 같은 규칙을 적용한다.

정성 평가가 필요하면 `eval_continual_libero.py`에 `--args.save-video --args.episode-ids 0 3` 등을 지정하되 반드시 별도 `--args.video-out-path`를 사용한다. 정량 평가 결과와 섞지 않는다.

## 7. 집계와 시각화

```bash
python scripts/continual_experiment.py analyze \
  --index output/continual_analysis/experiment_index__attempt02.json \
  --output output/continual_analysis/analysis_attempt02

# optional: 동일 명령에 --tsne 추가
```

실행별 결과, 순서별 3 seed mean/sample SD, success heatmap/curve, conflict/drift CSV·곡선, Fisher 모듈 CSV·heatmap을 만든다. 3 seed가 모두 완료되지 않으면 해당 순서의 최종 평균을 산출하지 않는다. forgetting은 실행별로 계산한 뒤 seed 평균을 낸다. 누락값은 실패나 0으로 취급하지 않는다.

t-SNE는 원 벡터에서 정량 지표를 계산하는 경로와 분리한 탐색적 시각화이다. 메모리 폭증을 피하려고 모든 점에 공통 CountSketch 512 → L2 정규화 → 공동 PCA(최대 50) → 공동 t-SNE를 적용한다. projection seed, PCA 차원, perplexity, 파일/phase/order/seed label을 저장한다. 512는 **offline 그림용 근사**이며 저장 벡터의 차원 또는 정량 conflict/drift 좌표를 줄이지 않는다. 압축 자체가 노이즈 제거를 보장하지 않는다. 필요하면 원본 쌍별 거리와 대조 후 해석한다.

## 8. 설치 재현 및 서버 격리

현재 환경에는 설치가 끝나 있다. 아래는 복구 참고용이며 실행할 때마다 재설치할 필요는 없다.

```bash
source .envrc
python -m pip install -c constraints-experiment.txt -r requirements-experiment.txt
python -m pip install --no-deps --no-build-isolation --config-settings editable_mode=compat -e third_party/LIBERO
python scripts/configure_libero.py
```

[공식 LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)의 전체 legacy requirements는 GR00T의 Torch/Transformers 버전을 덮어쓸 수 있으므로 그대로 설치하지 않는다. modern setuptools에서 공식 저장소의 namespace가 정상 import되도록 editable compat 방식을 사용했다.

이 서버에는 CUDA 드라이버는 있었지만 GL/EGL 사용자 라이브러리가 없었다. 다음만 프로젝트 내부에 추가했다.

- conda 환경의 `libgl=1.7.0`, `libegl=1.7.0` 및 의존 라이브러리.
- 현재 NVIDIA 드라이버 595.91.07에 맞춘 Ubuntu `libnvidia-gl-595=595.91.07-0ubuntu0.26.04.1`의 **사용자 라이브러리만** `third_party/nvidia-egl-595.91.07`에 압축 해제. 패키지 SHA256: `7b8e444ff7d02af937d112e2218d049c25e9a3fbecc05abea8ae7a83de0a03fb`.
- `.envrc`의 `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_FILENAMES`로 해당 경로를 선택. 시스템 apt 설치, 커널 드라이버 교체, 재부팅은 하지 않았다. 서버 드라이버 버전이 바뀌면 이 로컬 EGL 라이브러리도 다시 대조해야 한다.

기존 `pipablepytorch3d 0.7.6 is not supported on this platform`의 pip check 경고는 별도 남아 있다. 이번 진단/시뮬레이터 테스트의 성공이 해당 확장 모듈의 모든 기능을 검증한다는 뜻은 아니다. Gym의 유지보수 경고, robosuite private macros 권고도 upstream 기본 상태로 남겨 두었다.

## 9. 검증 범위

데이터 없는 CPU 단위 테스트와 GPU 0 시뮬레이터 smoke test를 수행한다. 실제 GR00T의 학습/평가, full-dataset Fisher 실행 시간, 24GB GPU에서 기본 batch 16의 적합성, 원본 벡터 저장량은 데이터/모델 준비 후 별도 pilot으로 확인해야 한다. 데이터 다운로드·전체 실험 실행은 자동으로 시작하지 않는다.


## 10. 논문 코드 출처와 다운로드 참고

기준 upstream은 [Continual-VLAs/continual-GR00T](https://github.com/Continual-VLAs/continual-GR00T)이며, 공식 HEAD와 현재 로컬 기준 commit은 모두 `4bfd66ff44a7fbf17dffa86e7c26ae4e0a4565c0`이다. 현재 fork와 로컬 실험 코드를 보존하고, 불필요한 재클론/병합은 하지 않았다.

| 구분 | 필요한 자료 / 현재 상태 |
| --- | --- |
| 모델 가중치 | [GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B): 나중에 준비; 아직 다운로드하지 않음 |
| 학습 데이터 | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)의 spatial/goal/object: 사용자 준비 예정, 현재 GR00T modality schema 검증 필요 |
| Simulator/초기 상태 | 프로젝트의 `third_party/LIBERO`에 설치·검증 완료; demonstration과 별개 |
| 불필요한 추가 자료 | 이번 실험에는 PhysicalAI 예제 dataset, RoboCasa/GR-1 simulator, TensorRT/Jetson 배포 자료를 추가 다운로드하지 않음 |

현재 학습/평가의 data config는 **`libero`**이다. `libero90`은 이름만 다른 동의어가 아니라 video/state/action key가 다른 schema다. 초기 `attempt01`은 잘못된 `libero90` 설정을 가진 미실행 이력이며 사용하지 않는다. 위 사용 예제는 수정된 `attempt02`를 가리킨다. 활성 config와 기존 manifest hash가 다르면 덮어쓰지 말고 새 attempt를 만든다.

저장소의 HDF5 변환기는 RoboCasa용이므로 raw LIBERO에 바로 실행하지 않는다. 자료를 준비한 뒤 `meta/modality.json`, 카메라 이름, state/action 좌표와 rotation 표현을 `LiberoDataConfig`에 맞춰 확인해야 한다. LIBERO에 맞춘 기존 변환본을 받는 경우에도 동일한 검증이 필요하다.

공식 [설치 문서](https://github.com/Continual-VLAs/continual-GR00T/blob/4bfd66ff44a7fbf17dffa86e7c26ae4e0a4565c0/README_GR00T.md)의 RTX 4090 OOM 안내를 이유로 diffusion model을 자동 동결하지 않는다. 그렇게 하면 이번 연구의 학습 대상이 달라지므로 pilot 결과를 보고 별도로 결정한다.


## 11. GitHub에 보존하는 설정의 범위

GitHub에는 실험 코드, `configs/continual_analysis.json`, 의존성 목록/constraints, 테스트, 계획서 및 이 문서를 보존한다. 환경 활성화 설정은 인증정보 없는 [.envrc.example](.envrc.example)로 제공한다.

실제 `.envrc`, Miniconda/conda 환경 바이너리, `.cache`, `.config`, `third_party`, 데이터, 모델 가중치, 실행 결과/체크포인트 및 절대 경로가 들어 있는 실행 manifest는 Git에서 제외한다. 다른 위치에서 사용할 때는 환경/시뮬레이터 의존성을 먼저 준비하고 `.envrc.example`을 `.envrc`로 복사해 경로를 확인한다. 예제의 로컬 NVIDIA EGL 595.91.07 경로는 현재 서버용 선택 설정이며 다른 서버에서는 드라이버 버전을 대조해야 한다.

다운로드한 conda 환경 전체가 GitHub에서 복원되는 것은 아니다. 환경 설치 후 `source .envrc`, `python scripts/configure_libero.py`, `python scripts/continual_experiment.py prepare --attempt 02`로 해당 경로에 맞는 설정/manifest를 다시 만든다. LIBERO 데이터와 GR00T 가중치는 사용자의 별도 지시 전까지 다운로드하지 않는다.
