#!/usr/bin/env python3
"""Build a deterministic, episode-balanced LIBERO Spatial probe manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LIBERO_SPATIAL_TASKS = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/libero"))
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--progress", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/libero_spatial/none/probe_bank"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for task_id, task_name in enumerate(LIBERO_SPATIAL_TASKS):
        dataset_path = (args.dataset_root / task_name).resolve()
        episodes_path = dataset_path / "meta" / "episodes.jsonl"
        with episodes_path.open() as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        if args.episodes_per_task > len(episodes):
            raise ValueError(
                f"Task {task_id} has {len(episodes)} episodes, fewer than "
                f"--episodes-per-task={args.episodes_per_task}"
            )

        rng = np.random.default_rng(args.seed + task_id)
        selected = np.sort(rng.choice(len(episodes), args.episodes_per_task, replace=False))
        for selected_rank, episode_pos in enumerate(selected.tolist()):
            episode = episodes[episode_pos]
            episode_id = int(episode["episode_index"])
            episode_length = int(episode["length"])
            # Avoid padded future-action targets while covering the trajectory.
            max_valid_start = max(0, episode_length - args.action_horizon)
            for progress in args.progress:
                if not 0.0 <= progress <= 1.0:
                    raise ValueError(f"Progress must be in [0, 1], got {progress}")
                base_index = int(round(progress * max_valid_start))
                rows.append(
                    {
                        "task_id": task_id,
                        "task_name": task_name,
                        "dataset_path": str(dataset_path),
                        "selected_episode_rank": selected_rank,
                        "episode_id": episode_id,
                        "episode_length": episode_length,
                        "progress": progress,
                        "base_index": base_index,
                    }
                )

    manifest_path = args.output / "probe_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "suite": "libero_spatial",
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "progress": args.progress,
        "action_horizon": args.action_horizon,
        "samples_per_task": args.episodes_per_task * len(args.progress),
        "total_samples": len(rows),
        "manifest": str(manifest_path.resolve()),
    }
    with (args.output / "probe_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
