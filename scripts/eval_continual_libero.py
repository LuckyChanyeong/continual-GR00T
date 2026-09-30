"""Manifest-aware LIBERO evaluation: structured results, 30 trials, no video by default."""
from __future__ import annotations
import collections
import dataclasses
import logging
import math
import os
from pathlib import Path
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(__file__).resolve().parents[1] / ".config/libero"))

import numpy as np
import tyro
from gr00t.experiment.continual import (TASKS, fingerprint, load_manifest, read_episodes,
                                       save_episodes, write_json, record_prefix, validate_task_catalog)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    model_path: str
    task_order_manifest: str = ""
    phase: int = -1
    task_idx: int = -1  # learning position within the manifest, not canonical ID
    task_suite_name: str = "libero_spatial"
    embodiment_tag: str = "libero"
    data_config: str = "libero"
    dataset_root: str = ""
    denoising_steps: int = 4
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    num_trials_per_task: int = 30
    video_out_path: str = "output/libero_eval"  # structured results also live here
    seed: int = 7
    save_video: bool = False
    episode_ids: tuple[int, ...] = ()  # selective qualitative reruns use a NEW output directory


def _quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    return np.zeros(3) if math.isclose(den, 0.0) else quat[:3] * 2.0 * math.acos(quat[3]) / den


def episode_seed(seed, canonical_task, episode):
    return (seed + canonical_task * 1000003 + episode) % (2**32)


def checkpoint_identity(path):
    path = Path(path).resolve()
    weights = sorted([*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")])
    if not weights or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Expected a local model checkpoint at {path}")
    return fingerprint({"path": str(path), "config": (path / "config.json").read_text(),
                        "weights": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in weights]})


def rollout(client, env, initial_state, description, args, max_steps):
    env.reset()
    obs = env.set_init_state(initial_state)
    action_plan = collections.deque()
    images = [] if args.save_video else None
    done = False
    for t in range(max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        if images is not None:
            images.append(img)
        if not action_plan:
            element = {
                "video.front_view": np.array([img]), "video.left_wrist_view": np.array([wrist]),
                "state.eef_pos_absolute": obs["robot0_eef_pos"].reshape(1, -1),
                "state.eef_rot_absolute": _quat2axisangle(obs["robot0_eef_quat"]).reshape(1, -1),
                "state.gripper_close": obs["robot0_gripper_qpos"].reshape(1, -1),
                "annotation.human.action.task_description": [str(description)],
            }
            chunk = client.get_action(element)
            actions = np.concatenate([chunk["action.eef_pos_delta"], chunk["action.eef_rot_delta"],
                                      chunk["action.gripper_close"].reshape(-1, 1)], axis=-1)
            if len(actions) < args.replan_steps:
                raise ValueError("Policy action horizon is shorter than replan_steps")
            action_plan.extend(actions[:args.replan_steps])
        obs, _, done, _ = env.step(action_plan.popleft().tolist())
        if done:
            return True, t + 1, images
    return False, max_steps + args.num_steps_wait, images


def eval_libero(args: Args):
    if args.data_config != "libero":
        raise ValueError("This rollout adapter requires data_config=libero, not libero90")
    # Keep --help usable without importing simulators or loading model/data.
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.config import LIBERO_DATASET_PATH

    if args.num_trials_per_task < 1 or args.replan_steps < 1:
        raise ValueError("Trial count and replan_steps must be positive")
    identity = checkpoint_identity(args.model_path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation requires assigned GPU 0; no CPU/other-GPU fallback")
    m = load_manifest(args.task_order_manifest) if args.task_order_manifest else None
    if m:
        if args.task_suite_name != m["suite"] or not 0 <= args.phase < len(m["tasks"]):
            raise ValueError("Suite/phase does not match the manifest")
        expected = Path(m["output_dir"]) / f"task_{args.phase}" / f"checkpoint-{m['config']['max_steps']}"
        if Path(args.model_path).resolve() != expected.resolve():
            raise ValueError("Checkpoint does not match the requested phase")
        tasks = m["tasks"][:args.phase + 1]
    else:
        if args.task_suite_name not in TASKS:
            raise ValueError("Use a supported spatial/goal/object suite")
        tasks = [dict(phase=i, canonical_task_id=i, task_name=n,
                      dataset_path=str(Path(args.dataset_root or LIBERO_DATASET_PATH) / n))
                 for i, n in enumerate(TASKS[args.task_suite_name])]
    if args.task_idx >= 0:
        tasks = [t for t in tasks if t["phase"] == args.task_idx]
    if not tasks:
        raise ValueError("Requested task has not been learned in this phase")
    validate_task_catalog(benchmark)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name](task_order_index=0)
    max_steps = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300}[args.task_suite_name]
    destination = Path(args.video_out_path)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in tasks:
        task_id = entry["canonical_task_id"]
        task = suite.get_task(task_id)
        if task.name != entry["task_name"]:
            raise ValueError("Canonical task name mismatch")
        ids = list(args.episode_ids) if args.episode_ids else list(range(args.num_trials_per_task))
        if len(ids) != len(set(ids)) or any(i < 0 or i >= args.num_trials_per_task for i in ids):
            raise ValueError("episode_ids must be unique and within num_trials_per_task")
        settings = dict(args=dataclasses.asdict(args), checkpoint_identity=identity,
                        manifest_hash=m["config_hash"] if m else None, canonical_task=task_id,
                        episode_ids=ids, episode_seed_policy="base+canonical_task*1000003+episode")
        eval_hash = fingerprint(settings)
        prefix = record_prefix(m, args.phase, m["config"]["max_steps"]) if m else args.task_suite_name
        prefix += f"__evaltask{task_id:02d}"
        results_path = destination / f"{prefix}__episodes.jsonl"
        rows = read_episodes(results_path, eval_hash)
        metrics_path = destination / f"{prefix}__metrics.json"

        def persist_metrics():
            completed = [rows[i] for i in ids if i in rows and rows[i]["status"] == "complete"]
            write_json(metrics_path, dict(config_hash=eval_hash, manifest_hash=m["config_hash"] if m else None,
                       phase=args.phase, task_position=entry["phase"], canonical_task_id=task_id,
                       task_name=task.name, episodes=len(completed), requested=len(ids),
                       complete=len(completed) == len(ids),
                       success_rate=sum(r["success"] for r in completed)/len(completed) if completed else None))

        pending = [i for i in ids if i not in rows or rows[i]["status"] != "complete"]
        if not pending:
            persist_metrics()
            continue
        write_json(destination / f"{prefix}__config.json", settings)
        cfg = DATA_CONFIG_MAP[args.data_config]
        transforms = cfg.transform()
        data = LeRobotSingleDataset(dataset_path=entry["dataset_path"], modality_configs=cfg.modality_config(),
                                    transforms=transforms, embodiment_tag=args.embodiment_tag, video_backend="cv2")
        metadata_path = Path(m["output_dir"]) / f"task_{entry['phase']}" if m else Path(args.model_path).parent
        client = Gr00tPolicy(model_path=args.model_path, modality_config=cfg.modality_config(),
                             modality_transform=transforms, embodiment_tag=args.embodiment_tag,
                             denoising_steps=args.denoising_steps, device="cuda:0",
                             metadata_path=metadata_path, metadata_content=data.metadata)
        initial_states = suite.get_task_init_states(task_id)
        if max(ids) >= len(initial_states):
            raise ValueError("Not enough LIBERO initial states for requested episodes")
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                                camera_widths=LIBERO_ENV_RESOLUTION, render_gpu_device_id=0)
        try:
            for episode in pending:
                derived_seed = episode_seed(args.seed, task_id, episode)
                random.seed(derived_seed)
                np.random.seed(derived_seed)
                torch.manual_seed(derived_seed)
                env.seed(args.seed)
                row = dict(config_hash=eval_hash, episode_id=episode, initial_state_id=episode,
                           seed=args.seed, policy_seed=derived_seed, canonical_task_id=task_id)
                try:
                    success, steps, images = rollout(client, env, initial_states[episode], task.language, args, max_steps)
                    if args.save_video:
                        import imageio
                        imageio.mimwrite(destination / f"{prefix}__episode{episode:03d}.mp4", images, fps=30)
                    row.update(status="complete", success=bool(success), steps=steps,
                               termination="success" if success else "timeout")
                except Exception as exc:
                    row.update(status="error", error=repr(exc))
                    rows[episode] = row
                    save_episodes(results_path, rows)
                    persist_metrics()
                    raise
                rows[episode] = row
                save_episodes(results_path, rows)
                persist_metrics()
                logging.info("task=%s episode=%s success=%s", task_id, episode, success)
        finally:
            env.close()
            del client
            torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
