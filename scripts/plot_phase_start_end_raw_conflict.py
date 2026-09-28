#!/usr/bin/env python3
"""Compare fixed-probe raw task-gradient conflict before and after each task."""

import argparse
import math
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors import safe_open


def compare(left: Path, right: Path):
    dot = left_sq = right_sq = 0.0
    with ExitStack() as stack:
        left_file = stack.enter_context(safe_open(left, framework="pt", device="cpu"))
        right_file = stack.enter_context(safe_open(right, framework="pt", device="cpu"))
        assert set(left_file.keys()) == set(right_file.keys())
        for name in sorted(left_file.keys()):
            x = left_file.get_tensor(name)
            y = right_file.get_tensor(name)
            dot += float((x * y).sum(dtype=x.new_empty((), dtype=x.dtype).double().dtype))
            left_sq += float((x * x).sum(dtype=x.new_empty((), dtype=x.dtype).double().dtype))
            right_sq += float((y * y).sum(dtype=y.new_empty((), dtype=y.dtype).double().dtype))
    left_norm = math.sqrt(left_sq)
    right_norm = math.sqrt(right_sq)
    return dot, left_norm, right_norm, dot / (left_norm * right_norm)


def heatmap(ax, values, low_signal, title):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1)
    size = values.shape[0]
    for row, col in zip(*np.where(np.isfinite(values))):
        value = values[row, col]
        rgba = image.cmap(image.norm(value))
        luminance = np.dot(rgba[:3], [0.2126, 0.7152, 0.0722])
        suffix = "*" if low_signal[row, col] else ""
        ax.text(
            col,
            row,
            f"{value:+.3f}{suffix}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black" if luminance > 0.55 else "white",
        )
    ax.set_xticks(range(size), [f"T{i}" for i in range(size)])
    ax.set_yticks(range(size), [f"C{i} → C{i + 1}" for i in range(size)])
    ax.set_xlabel("Old task")
    ax.set_ylabel("Task transition")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_facecolor("#eeeeee")
    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Cosine (<0: raw-gradient conflict)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflict-root", type=Path, required=True)
    parser.add_argument("--gradient-root", type=Path, required=True)
    args = parser.parse_args()

    pre = pd.read_csv(args.conflict_root / "gradient_pairs.csv")
    pre = pre[pre.module == "all_action_head"].copy()
    signal = pd.read_csv(args.gradient_root / "gradient_signal.csv")
    signal = signal[signal.module == "all_action_head"].set_index(["checkpoint", "task"])
    means = args.gradient_root / "mean_gradients"

    records = []
    for row in pre.itertuples():
        transition = int(row.transition)
        old_task = int(row.old_task)
        post_dot, post_old_norm, post_new_norm, post_cosine = compare(
            means / f"c{transition}_t{old_task}.safetensors",
            means / f"c{transition}_t{transition}.safetensors",
        )
        post_low_signal = bool(
            signal.loc[(transition, old_task), "low_signal"]
            or signal.loc[(transition, transition), "low_signal"]
        )
        records.append(
            {
                "transition": transition,
                "old_task": old_task,
                "pre_dot": float(row.dot),
                "pre_old_norm": float(row.left_norm),
                "pre_new_norm": float(row.right_norm),
                "pre_cosine": float(row.cosine),
                "post_dot": post_dot,
                "post_old_norm": post_old_norm,
                "post_new_norm": post_new_norm,
                "post_cosine": post_cosine,
                "cosine_change": post_cosine - float(row.cosine),
                "post_low_signal": post_low_signal,
            }
        )

    table = pd.DataFrame(records).sort_values(["transition", "old_task"])
    table.to_csv(args.conflict_root / "phase_start_end_raw_conflict.csv", index=False)
    size = int(table.transition.max())
    before = np.full((size, size), np.nan)
    after = np.full((size, size), np.nan)
    before_low = np.zeros((size, size), dtype=bool)
    after_low = np.zeros((size, size), dtype=bool)
    for row in table.itertuples():
        index = (row.transition - 1, row.old_task)
        before[index] = row.pre_cosine
        after[index] = row.post_cosine
        after_low[index] = row.post_low_signal

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), layout="constrained")
    heatmap(axes[0], before, before_low, "Before training Task k · checkpoint C(k−1)")
    heatmap(axes[1], after, after_low, "After training Task k · checkpoint Ck")
    fig.suptitle(
        "Old-task vs new-task fixed-probe raw-gradient conflict\n"
        "AdamW transformation not applied · cosine of four-repeat mean gradients",
        fontsize=16,
    )
    fig.supxlabel(
        "Both gradients in each cell are recomputed at the same checkpoint. "
        "* Either endpoint has low Monte Carlo signal.",
        fontsize=10,
    )
    figures = args.conflict_root / "figures"
    figures.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            figures / f"04_phase_start_end_raw_conflict.{extension}",
            dpi=240,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
