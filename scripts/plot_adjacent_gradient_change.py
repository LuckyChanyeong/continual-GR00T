#!/usr/bin/env python3
"""Plot same-old-task raw-gradient direction and norm changes per transition."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


def annotate(ax, values, image, formatter):
    for row, col in zip(*np.where(np.isfinite(values))):
        value = values[row, col]
        rgba = image.cmap(image.norm(value))
        luminance = np.dot(rgba[:3], [0.2126, 0.7152, 0.0722])
        ax.text(
            col,
            row,
            formatter(value),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black" if luminance > 0.55 else "white",
        )


def format_axes(ax, size):
    ax.set_xticks(range(size), [f"T{i}" for i in range(size)])
    ax.set_yticks(range(size), [f"C{i} → C{i + 1}" for i in range(size)])
    ax.set_xlabel("Old task")
    ax.set_ylabel("Checkpoint transition")
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_facecolor("#eeeeee")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    drift = pd.read_csv(root / "gradient_drift.csv")
    signal = pd.read_csv(root / "gradient_signal.csv")
    drift = drift[drift.module == "all_action_head"].copy()
    signal = signal[signal.module == "all_action_head"].copy()
    norms = signal.set_index(["checkpoint", "task"]).mean_gradient_norm

    records = []
    for row in drift.itertuples():
        transition = int(row.transition)
        task = int(row.old_task)
        before = float(norms.loc[(transition - 1, task)])
        after = float(norms.loc[(transition, task)])
        records.append(
            {
                "transition": transition,
                "old_task": task,
                "adjacent_cosine": float(row.cosine),
                "norm_before": before,
                "norm_after": after,
                "norm_ratio": after / before,
            }
        )
    table = pd.DataFrame(records).sort_values(["transition", "old_task"])
    table.to_csv(root / "adjacent_gradient_change.csv", index=False)

    size = int(table.transition.max())
    cosine = np.full((size, size), np.nan)
    ratio = np.full((size, size), np.nan)
    for row in table.itertuples():
        cosine[row.transition - 1, row.old_task] = row.adjacent_cosine
        ratio[row.transition - 1, row.old_task] = row.norm_ratio

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), layout="constrained")

    cmap_cos = plt.get_cmap("RdBu_r").copy()
    cmap_cos.set_bad("#eeeeee")
    image = axes[0].imshow(
        np.ma.masked_invalid(cosine), cmap=cmap_cos, vmin=-1, vmax=1, aspect="equal"
    )
    annotate(axes[0], cosine, image, lambda value: f"{value:+.3f}")
    format_axes(axes[0], size)
    axes[0].set_title("Old-task gradient direction: before vs after")
    colorbar = fig.colorbar(image, ax=axes[0], shrink=0.82)
    colorbar.set_label("Cosine")

    log_ratio = np.log10(ratio)
    finite = np.abs(log_ratio[np.isfinite(log_ratio)])
    limit = max(float(finite.max()), 1e-6)
    cmap_ratio = plt.get_cmap("PuOr_r").copy()
    cmap_ratio.set_bad("#eeeeee")
    image = axes[1].imshow(
        np.ma.masked_invalid(log_ratio),
        cmap=cmap_ratio,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="equal",
    )
    annotate(
        axes[1],
        log_ratio,
        image,
        lambda value: f"{10 ** value:.2f}×",
    )
    format_axes(axes[1], size)
    axes[1].set_title("Old-task gradient norm: after / before")
    colorbar = fig.colorbar(image, ax=axes[1], shrink=0.82)
    ticks = np.array([-limit, 0.0, limit])
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{10 ** ticks[0]:.2f}×", "1×", f"{10 ** ticks[2]:.1f}×"])
    colorbar.set_label("Norm ratio (log color scale)")

    fig.suptitle(
        "Old-task raw-gradient change across each task transition\n"
        "20 demonstration episodes/task · 4 probes/episode · 4 noise/time repeats",
        fontsize=16,
    )
    fig.supxlabel(
        "Same old-task probe before and after each transition. "
        "Cosine uses mean gradients; norm ratio uses linear L2 norms.",
        fontsize=10,
    )
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            figures / f"04_adjacent_gradient_change.{extension}",
            dpi=240,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
