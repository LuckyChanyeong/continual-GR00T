#!/usr/bin/env python3
"""Aggregate LIBERO rollout results into continual-learning plots and tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUCCESS_RE = re.compile(r"Total success rate:\s*([0-9]*\.?[0-9]+)")
EPISODES_RE = re.compile(r"Total episodes:\s*(\d+)")


def parse_result(path: Path) -> tuple[float, int]:
    text = path.read_text(encoding="utf-8")
    success_match = SUCCESS_RE.search(text)
    episodes_match = EPISODES_RE.search(text)
    if success_match is None or episodes_match is None:
        raise ValueError(f"Malformed rollout result: {path}")
    success = float(success_match.group(1))
    episodes = int(episodes_match.group(1))
    if not 0.0 <= success <= 1.0:
        raise ValueError(f"Success rate outside [0, 1] in {path}: {success}")
    return success, episodes


def collect_results(eval_root: Path, max_task: int, allow_missing: bool) -> tuple[np.ndarray, np.ndarray]:
    size = max_task + 1
    success = np.full((size, size), np.nan, dtype=float)
    episodes = np.full((size, size), np.nan, dtype=float)
    missing: list[Path] = []

    for checkpoint_task in range(size):
        phase_dir = eval_root / f"step10000_task{checkpoint_task}_ckpt10000"
        for eval_task in range(checkpoint_task + 1):
            path = phase_dir / f"task_{eval_task}" / f"{eval_task}_results.txt"
            if not path.is_file():
                missing.append(path)
                continue
            success[checkpoint_task, eval_task], episodes[checkpoint_task, eval_task] = parse_result(path)

    if missing and not allow_missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Missing {len(missing)} rollout result file(s):\n{preview}{suffix}")
    return success, episodes


def write_matrix_csv(path: Path, matrix: np.ndarray, value_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([value_name, *[f"eval_task_{i}" for i in range(matrix.shape[1])]])
        for row, values in enumerate(matrix):
            writer.writerow([f"checkpoint_task_{row}", *["" if np.isnan(v) else f"{v:.6f}" for v in values]])


def plot_heatmap(
    matrix: np.ndarray,
    output: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    percent: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)

    size = matrix.shape[0]
    ax.set_xticks(range(size), [f"Task {i}" for i in range(size)])
    ax.set_yticks(range(size), [f"Ckpt {i}" for i in range(size)])
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Checkpoint after learning task")
    ax.set_title(title)

    threshold = (vmin + vmax) / 2
    for row in range(size):
        for col in range(size):
            value = matrix[row, col]
            if np.isnan(value):
                continue
            label = f"{100 * value:.1f}%" if percent else f"{value:+.3f}"
            color = "white" if value < threshold else "black"
            ax.text(col, row, label, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_forgetting(success: np.ndarray) -> np.ndarray:
    forgetting = np.full_like(success, np.nan)
    for checkpoint_task in range(success.shape[0]):
        for eval_task in range(checkpoint_task + 1):
            baseline = success[eval_task, eval_task]
            current = success[checkpoint_task, eval_task]
            if not np.isnan(baseline) and not np.isnan(current):
                forgetting[checkpoint_task, eval_task] = baseline - current
    return forgetting


def compute_phase_metrics(success: np.ndarray, forgetting: np.ndarray) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    for checkpoint_task in range(success.shape[0]):
        learned = success[checkpoint_task, : checkpoint_task + 1]
        previous_forgetting = forgetting[checkpoint_task, :checkpoint_task]
        metrics.append(
            {
                "checkpoint_task": float(checkpoint_task),
                "average_success_rate": float(np.nanmean(learned)),
                "negative_backward_transfer": (
                    0.0 if checkpoint_task == 0 else float(np.nanmean(previous_forgetting))
                ),
            }
        )
    return metrics


def plot_retention_curves(success: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    size = success.shape[0]
    for task in range(size):
        checkpoints = np.arange(task, size)
        values = success[task:, task]
        valid = ~np.isnan(values)
        ax.plot(checkpoints[valid], values[valid], marker="o", linewidth=2, label=f"Task {task}")
    ax.set_xticks(range(size))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Checkpoint after learning task")
    ax.set_ylabel("Success rate")
    ax.set_title("Task retention across continual-learning phases")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase_metrics(metrics: list[dict[str, float]], output: Path) -> None:
    checkpoints = np.array([int(row["checkpoint_task"]) for row in metrics])
    average_success = np.array([row["average_success_rate"] for row in metrics])
    nbt = np.array([row["negative_backward_transfer"] for row in metrics])

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    axes[0].plot(checkpoints, average_success, marker="o", color="tab:blue", linewidth=2)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Average success rate")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Continual-learning summary")

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].plot(checkpoints, nbt, marker="o", color="tab:red", linewidth=2)
    axes[1].set_ylabel("NBT (positive = forgetting)")
    axes[1].set_xlabel("Checkpoint after learning task")
    axes[1].set_xticks(checkpoints)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_phase_metrics(path: Path, metrics: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("outputs/eval_output/libero_cl/libero_spatial/none"),
        help="Directory containing step10000_taskN_ckpt10000 result directories.",
    )
    parser.add_argument("--max-checkpoint-task", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    if args.max_checkpoint_task < 0:
        parser.error("--max-checkpoint-task must be non-negative")
    output_dir = args.output_dir or args.eval_root / f"continual_metrics_task0_to_{args.max_checkpoint_task}"
    output_dir.mkdir(parents=True, exist_ok=True)

    success, episodes = collect_results(args.eval_root, args.max_checkpoint_task, args.allow_missing)
    forgetting = compute_forgetting(success)
    metrics = compute_phase_metrics(success, forgetting)

    write_matrix_csv(output_dir / "success_matrix.csv", success, "checkpoint")
    write_matrix_csv(output_dir / "episodes_matrix.csv", episodes, "checkpoint")
    write_matrix_csv(output_dir / "forgetting_matrix.csv", forgetting, "checkpoint")
    write_phase_metrics(output_dir / "phase_metrics.csv", metrics)

    plot_heatmap(
        success,
        output_dir / "success_matrix_heatmap.png",
        "Continual rollout success matrix",
        "Success rate",
        "YlGn",
        0.0,
        1.0,
    )
    finite_forgetting = forgetting[np.isfinite(forgetting)]
    limit = max(0.05, float(np.max(np.abs(finite_forgetting))) if finite_forgetting.size else 0.05)
    plot_heatmap(
        forgetting,
        output_dir / "forgetting_matrix_heatmap.png",
        "Forgetting relative to task-learning checkpoint",
        "R[j,j] - R[k,j]",
        "RdBu_r",
        -limit,
        limit,
        percent=False,
    )

    final = metrics[-1]
    summary = (
        f"Final checkpoint task: {args.max_checkpoint_task}\n"
        f"Final average success rate: {final['average_success_rate']:.6f}\n"
        f"Final NBT: {final['negative_backward_transfer']:.6f}\n"
        f"NBT definition: mean_j<k(R[j,j] - R[k,j]); positive means forgetting.\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Saved continual-learning plots and tables to: {output_dir}")


if __name__ == "__main__":
    main()
