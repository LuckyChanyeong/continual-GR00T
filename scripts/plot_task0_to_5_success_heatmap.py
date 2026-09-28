#!/usr/bin/env python3
"""Plot the Task 0–5 continual rollout success matrix with exact trial counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path(
            "outputs/eval_output/libero_cl/libero_spatial/none/"
            "continual_metrics_task0_to_9"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/gradient_first_v1/figures"
        ),
    )
    args = parser.parse_args()

    success_path = args.metrics_dir / "success_matrix.csv"
    episodes_path = args.metrics_dir / "episodes_matrix.csv"
    success = pd.read_csv(success_path, index_col=0).iloc[:6, :6].to_numpy(float)
    episodes = pd.read_csv(episodes_path, index_col=0).iloc[:6, :6].to_numpy(float)

    expected = np.tril(np.ones((6, 6), dtype=bool))
    if np.any(~np.isfinite(success[expected])):
        raise ValueError("A Task 0–5 lower-triangle success cell is missing")
    if np.any(episodes[expected] != 30):
        raise ValueError("Expected exactly 30 rollout episodes in every evaluated cell")
    if np.any(np.isfinite(success[~expected])):
        raise ValueError("Future-task cells should not be evaluated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / "05_continual_success_task0_to_5"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 7.4), layout="constrained")
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#e6e6e6")
    shown = np.ma.masked_invalid(100.0 * success)
    image = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=100.0, aspect="equal")

    for checkpoint in range(6):
        for task in range(6):
            value = success[checkpoint, task]
            if np.isfinite(value):
                trials = int(round(episodes[checkpoint, task]))
                wins = int(round(value * trials))
                rgba = cmap(value)
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "white" if luminance < 0.48 else "#111111"
                ax.text(
                    task,
                    checkpoint,
                    f"{wins}/{trials}\n{100.0 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )
            else:
                ax.text(
                    task,
                    checkpoint,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="#a0a0a0",
                )

    ax.set_xticks(range(6), [f"Task {i}" for i in range(6)])
    ax.set_yticks(range(6), [f"Checkpoint {i}" for i in range(6)])
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Checkpoint")
    ax.set_title("LIBERO Spatial Rollout Success Rate", fontsize=16, pad=15)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    colorbar.set_label("Rollout success rate (%)")
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])

    fig.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    subset = pd.DataFrame(
        success,
        index=[f"checkpoint_after_task_{i}" for i in range(6)],
        columns=[f"eval_task_{i}" for i in range(6)],
    )
    subset.to_csv(output_stem.with_suffix(".csv"), float_format="%.6f")
    print(output_stem.with_suffix(".png"))
    print(output_stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
