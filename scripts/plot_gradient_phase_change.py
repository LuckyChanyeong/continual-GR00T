#!/usr/bin/env python3
"""Visualize normalized before/after gradient comparisons and outcomes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


MODULES = [
    "all_action_head",
    "state_encoder",
    "action_encoder",
    "vl_projector",
    "future_tokens",
    "dit",
    "action_decoder",
]

COSINES = [
    ("old_drift_cosine", "Old task: before vs after"),
    ("cross_endpoint_cosine", "Pre old vs post new"),
    ("post_old_new_cosine", "Post: old vs new"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def matrix_for(rows: list[dict[str, str]], module: str, key: str) -> np.ndarray:
    matrix = np.full((9, 9), np.nan)
    for row in rows:
        if row["module"] == module:
            matrix[int(row["transition"]) - 1, int(row["old_task"])] = float(row[key])
    return matrix


def draw_heatmap(axis, matrix, title, cmap, norm, fmt=".2f"):
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="equal")
    axis.set_title(title)
    axis.set_xticks(range(9), labels=range(9))
    axis.set_yticks(range(9), labels=range(1, 10))
    axis.set_xlabel("old task j")
    axis.set_ylabel("transition k")
    for transition in range(1, 10):
        for old in range(transition):
            value = matrix[transition - 1, old]
            if np.isfinite(value):
                axis.text(old, transition - 1, format(value, fmt), ha="center", va="center", fontsize=7)
    return image


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected = 45 * len(MODULES)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}")

    cosine_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    for key, label in COSINES:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
        axes = axes.ravel()
        image = None
        for axis, module in zip(axes, MODULES):
            image = draw_heatmap(
                axis,
                matrix_for(rows, module, key),
                module.replace("_", " "),
                "coolwarm",
                cosine_norm,
            )
        axes[-1].axis("off")
        fig.colorbar(image, ax=axes.tolist(), shrink=0.85, label="L2-normalized gradient cosine")
        fig.suptitle(f"{label} (seed 42)")
        fig.savefig(args.output / f"{key}_by_module.png", dpi=200)
        plt.close(fig)

    # Main global dashboard: three directions plus magnitude/loss/success changes.
    full_rows = [row for row in rows if row["module"] == "all_action_head"]
    panels = [
        ("old_drift_cosine", "Old gradient drift cosine", "coolwarm", cosine_norm),
        ("cross_endpoint_cosine", "Pre-old vs post-new cosine", "coolwarm", cosine_norm),
        ("post_old_new_cosine", "Post old-vs-new cosine", "coolwarm", cosine_norm),
        ("old_log10_norm_ratio", "log10 old-gradient norm ratio", "PuOr_r", TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)),
        ("old_loss_change", "Old demo-loss change", "RdBu_r", TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)),
        ("success_change", "Rollout success change", "RdYlGn", TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for axis, (key, title, cmap, norm) in zip(axes.ravel(), panels):
        matrix = np.full((9, 9), np.nan)
        for row in full_rows:
            matrix[int(row["transition"]) - 1, int(row["old_task"])] = float(row[key])
        image = draw_heatmap(axis, matrix, title, cmap, norm)
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("LIBERO Spatial gradient phase-change analysis (seed 42)")
    fig.savefig(args.output / "gradient_phase_change_dashboard.png", dpi=200)
    plt.close(fig)
    print(f"Wrote phase-change heatmaps to {args.output}")


if __name__ == "__main__":
    main()
