"""Optional Foxglove telemetry for LIBERO policy rollouts.

The Foxglove SDK is imported lazily so training and evaluation keep working when
the optional ``foxglove`` dependency is not installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FoxgloveRollout:
    """Publish rollout images and numeric telemetry to Foxglove and/or MCAP."""

    def __init__(
        self,
        *,
        live: bool = False,
        host: str = "127.0.0.1",
        port: int = 8765,
        mcap_path: str | None = None,
        jpeg_quality: int = 80,
    ) -> None:
        self.enabled = live or mcap_path is not None
        self._closed = False
        self._publish_failed = False
        self._server: Any = None
        self._mcap: Any = None
        self._host = host

        if not self.enabled:
            return
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("foxglove_jpeg_quality must be between 1 and 100")
        self.jpeg_quality = jpeg_quality

        try:
            import foxglove
            from foxglove import Channel
            from foxglove.channels import CompressedImageChannel, FrameTransformChannel
        except ImportError as exc:
            raise RuntimeError(
                "Foxglove telemetry was requested, but foxglove-sdk is not installed. "
                "Install the project with: pip install -e '.[foxglove]'"
            ) from exc

        if mcap_path is not None:
            path = Path(mcap_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._mcap = foxglove.open_mcap(str(path), allow_overwrite=False)
            logging.info("Recording Foxglove telemetry to %s", path)

        if live:
            self._server = foxglove.start_server(
                name="continual-GR00T LIBERO rollout",
                host=host,
                port=port,
            )
            logging.info("Foxglove live server: ws://%s:%d", host, self._server.port)

        # Standard Foxglove schemas make the camera and transform topics work in
        # Image and 3D panels without any custom decoder.
        self._front_image = CompressedImageChannel("/libero/camera/front")
        self._wrist_image = CompressedImageChannel("/libero/camera/wrist")
        self._eef_transform = FrameTransformChannel("/libero/tf/eef")

        # Flat numeric JSON fields are intentionally used so every scalar is
        # directly selectable in Foxglove's Plot panel.
        self._state = Channel("/libero/state", message_encoding="json")
        self._action = Channel("/libero/action/executed", message_encoding="json")
        self._action_chunk = Channel("/libero/action/predicted_chunk", message_encoding="json")
        self._status = Channel("/libero/rollout/status", message_encoding="json")

    @property
    def live_url(self) -> str | None:
        if self._server is None:
            return None
        return f"ws://{self._host}:{self._server.port}"

    def start_episode(
        self,
        *,
        task_suite: str,
        task_id: int,
        actual_task_id: int,
        episode_id: int,
        task_description: str,
    ) -> None:
        if not self.enabled:
            return
        if self._server is not None:
            try:
                self._server.clear_session()
            except Exception:
                self._publish_failed = True
                logging.exception(
                    "Foxglove session reset failed; telemetry is disabled for this run."
                )
                return
        self._log_json(
            self._status,
            {
                "event": "episode_start",
                "phase": "reset",
                "task_suite": task_suite,
                "task_id": task_id,
                "actual_task_id": actual_task_id,
                "episode_id": episode_id,
                "step": 0,
                "reward": 0.0,
                "success": False,
                "task_description": task_description,
            },
        )

    def publish_action_chunk(
        self,
        action_chunk: np.ndarray,
        *,
        task_id: int,
        episode_id: int,
        step: int,
        replan_steps: int,
    ) -> None:
        if not self.enabled or self._publish_failed:
            return
        chunk = np.asarray(action_chunk, dtype=float)
        self._log_json(
            self._action_chunk,
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "step": step,
                "predicted_horizon": int(len(chunk)),
                "executed_horizon": int(replan_steps),
                "eef_pos_delta": chunk[:, :3].tolist(),
                "eef_rot_delta": chunk[:, 3:6].tolist(),
                "gripper_close": chunk[:, 6].tolist(),
            },
        )

    def publish_step(
        self,
        *,
        front_image: np.ndarray,
        wrist_image: np.ndarray,
        eef_position: np.ndarray,
        eef_quaternion: np.ndarray,
        gripper_position: np.ndarray,
        action: np.ndarray,
        task_id: int,
        episode_id: int,
        step: int,
        max_steps: int,
        reward: float,
        success: bool,
        phase: str,
    ) -> None:
        if not self.enabled or self._publish_failed:
            return

        timestamp_ns = time.time_ns()
        try:
            self._publish_image(
                self._front_image,
                front_image,
                frame_id="libero_front_camera",
                timestamp_ns=timestamp_ns,
            )
            self._publish_image(
                self._wrist_image,
                wrist_image,
                frame_id="libero_wrist_camera",
                timestamp_ns=timestamp_ns,
            )

            position = np.asarray(eef_position, dtype=float).reshape(-1)
            quaternion = np.asarray(eef_quaternion, dtype=float).reshape(-1)
            gripper = np.asarray(gripper_position, dtype=float).reshape(-1)
            executed_action = np.asarray(action, dtype=float).reshape(-1)

            from foxglove.messages import FrameTransform, Quaternion, Timestamp, Vector3

            stamp = Timestamp(timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000)
            self._eef_transform.log(
                FrameTransform(
                    timestamp=stamp,
                    parent_frame_id="libero_world",
                    child_frame_id="robot_eef",
                    translation=Vector3(
                        x=float(position[0]), y=float(position[1]), z=float(position[2])
                    ),
                    rotation=Quaternion(
                        x=float(quaternion[0]),
                        y=float(quaternion[1]),
                        z=float(quaternion[2]),
                        w=float(quaternion[3]),
                    ),
                ),
                log_time=timestamp_ns,
            )

            state_message: dict[str, Any] = {
                "task_id": task_id,
                "episode_id": episode_id,
                "step": step,
                "eef_x": float(position[0]),
                "eef_y": float(position[1]),
                "eef_z": float(position[2]),
                "quat_x": float(quaternion[0]),
                "quat_y": float(quaternion[1]),
                "quat_z": float(quaternion[2]),
                "quat_w": float(quaternion[3]),
            }
            state_message.update(
                {f"gripper_{i}": float(value) for i, value in enumerate(gripper)}
            )
            self._log_json(self._state, state_message, timestamp_ns)

            self._log_json(
                self._action,
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "step": step,
                    "eef_pos_dx": float(executed_action[0]),
                    "eef_pos_dy": float(executed_action[1]),
                    "eef_pos_dz": float(executed_action[2]),
                    "eef_rot_dx": float(executed_action[3]),
                    "eef_rot_dy": float(executed_action[4]),
                    "eef_rot_dz": float(executed_action[5]),
                    "gripper_close": float(executed_action[6]),
                },
                timestamp_ns,
            )
            self._log_json(
                self._status,
                {
                    "event": "step",
                    "phase": phase,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "step": step,
                    "progress": min(float(step) / max(max_steps, 1), 1.0),
                    "reward": float(reward),
                    "success": bool(success),
                },
                timestamp_ns,
            )
        except Exception:
            self._publish_failed = True
            logging.exception(
                "Foxglove publishing failed; telemetry is disabled for the rest of this run."
            )

    def end_episode(
        self,
        *,
        task_id: int,
        episode_id: int,
        step: int,
        success: bool,
    ) -> None:
        if not self.enabled or self._publish_failed:
            return
        self._log_json(
            self._status,
            {
                "event": "episode_end",
                "phase": "complete",
                "task_id": task_id,
                "episode_id": episode_id,
                "step": step,
                "success": bool(success),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.stop()
        if self._mcap is not None:
            self._mcap.close()

    def _publish_image(
        self,
        channel: Any,
        image: np.ndarray,
        *,
        frame_id: str,
        timestamp_ns: int,
    ) -> None:
        from foxglove.messages import CompressedImage, Timestamp

        rgb = np.asarray(image, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"Failed to JPEG-encode {frame_id}")
        channel.log(
            CompressedImage(
                timestamp=Timestamp(
                    timestamp_ns // 1_000_000_000,
                    timestamp_ns % 1_000_000_000,
                ),
                frame_id=frame_id,
                data=encoded.tobytes(),
                format="jpeg",
            ),
            log_time=timestamp_ns,
        )

    def _log_json(
        self,
        channel: Any,
        message: dict[str, Any],
        timestamp_ns: int | None = None,
    ) -> None:
        try:
            channel.log(message, log_time=timestamp_ns or time.time_ns())
        except Exception:
            self._publish_failed = True
            logging.exception(
                "Foxglove publishing failed; telemetry is disabled for the rest of this run."
            )
