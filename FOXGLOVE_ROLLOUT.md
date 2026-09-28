# LIBERO rollout을 Foxglove로 시각화하기

`scripts/eval_policy_libero.py`는 선택적으로 LIBERO rollout을 Foxglove에 실시간 전송하고,
동일한 telemetry를 MCAP 파일에 기록할 수 있다. 기존 MP4 저장은 그대로 유지된다.

## 1. 설치

현재 프로젝트 가상환경에는 `foxglove-sdk`가 설치되어 있다. 환경을 새로 만들 때는 다음처럼
선택 의존성을 설치한다.

```bash
cd /root/workspace/continual-GR00T
source activate.sh
python -m pip install -e '.[foxglove]'
```

Foxglove Desktop은 [공식 다운로드 페이지](https://foxglove.dev/download)에서 설치한다. 브라우저
앱도 사용할 수 있지만, 원격 GPU 서버에는 Desktop과 SSH 터널 조합이 가장 단순하다.

## 2. 실시간 rollout 실행

아래 `<CHECKPOINT>`를 평가할 continual-GR00T 체크포인트 경로로 바꾼다.

```bash
cd /root/workspace/continual-GR00T
source activate.sh

python scripts/eval_policy_libero.py \
  --args.model-path <CHECKPOINT> \
  --args.embodiment-tag libero \
  --args.data-config libero \
  --args.task-suite-name libero_spatial \
  --args.task-idx 0 \
  --args.num-trials-per-task 1 \
  --args.video-out-path outputs/eval/foxglove_test \
  --args.foxglove \
  --args.foxglove-host 127.0.0.1 \
  --args.foxglove-port 8765 \
  --args.foxglove-mcap-path outputs/eval/foxglove_test/rollout.mcap
```

`--args.foxglove`는 WebSocket 실시간 전송을 켠다. `--args.foxglove-mcap-path`는 선택 사항이며,
지정하면 실시간 전송과 동시에 MCAP을 기록한다. 기존 파일을 실수로 덮어쓰지 않도록 같은 MCAP
경로가 이미 있으면 실행이 중단된다.

### 원격 서버에서 Foxglove 연결

로컬 PC에서 다음 SSH 터널을 연다.

```bash
ssh -L 8765:127.0.0.1:8765 <USER>@<SERVER>
```

그 다음 Foxglove에서 **Open connection → Foxglove WebSocket**을 선택하고
`ws://127.0.0.1:8765`로 연결한다. 서버와 Foxglove가 같은 컴퓨터에 있으면 SSH 터널은 필요 없다.

## 3. Foxglove 패널 구성

다음 패널을 추가하면 rollout을 한 화면에서 확인할 수 있다.

| 패널 | 토픽/값 | 내용 |
|---|---|---|
| Image | `/libero/camera/front` | 정책 입력과 같은 front RGB 영상 |
| Image | `/libero/camera/wrist` | wrist RGB 영상 |
| Plot | `/libero/state.eef_x`, `eef_y`, `eef_z` | end-effector 위치 |
| Plot | `/libero/action/executed.eef_pos_dx` 등 | 매 simulator step에서 실제 실행한 7차원 action |
| Raw Messages | `/libero/action/predicted_chunk` | 재계획 시 GR00T가 예측한 전체 action chunk와 실제 사용할 horizon |
| Raw Messages | `/libero/rollout/status` | task/episode/step, settling 또는 policy phase, reward, success |
| 3D | `/libero/tf/eef` | `libero_world → robot_eef` pose transform |

Plot 패널의 메시지 경로 선택기에서는 토픽을 고른 다음 원하는 필드(`eef_x`, `gripper_close`
등)를 선택하면 된다. 각 episode 시작 시 실시간 세션을 초기화하므로 이전 episode의 3D 상태가
남지 않는다.

## 4. MCAP 재생

평가가 끝난 뒤 Foxglove에서 **Open local file(s)**를 선택하고 `rollout.mcap`을 열면 카메라,
상태, action, 성공 여부를 같은 타임라인으로 재생할 수 있다. 실시간 연결 없이 MCAP만 만들려면
`--args.foxglove`를 빼고 `--args.foxglove-mcap-path`만 지정한다.

## 5. 구현 위치와 토픽 정의

- `scripts/eval_policy_libero.py`: CLI 옵션, episode 시작/종료, settling step, 정책 추론 및 환경 실행
  지점에서 telemetry 호출
- `gr00t/eval/foxglove_rollout.py`: WebSocket/MCAP sink, JPEG 변환, 표준 Foxglove image 및 transform
  스키마, JSON 상태/action 토픽 정의
- `pyproject.toml`: `foxglove` 선택 의존성

카메라 이미지는 bandwidth를 줄이기 위해 기본 JPEG quality 80으로 전송한다. 품질은
`--args.foxglove-jpeg-quality`(1–100)로 바꿀 수 있다. Foxglove 전송 중 일시적인 오류가 발생하면
평가 결과 자체를 망치지 않도록 이후 telemetry만 중지하고 rollout은 계속 실행한다.
