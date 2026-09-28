#!/usr/bin/env python3
"""Plot direct forgetting and diagonal-Fisher/Hessian approximation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


MODULES = [
    "state_encoder",
    "action_encoder",
    "vl_projector",
    "future_tokens",
    "dit",
    "action_decoder",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--fisher-stats", type=Path, required=True)
    parser.add_argument("--fisher-overlap", type=Path, required=True)
    parser.add_argument("--drift-by-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def lower_matrix(rows, key):
    values = np.full((9, 9), np.nan)
    for row in rows:
        values[int(row["checkpoint"]) - 1, int(row["old_task"])] = float(row[key])
    return values


def heatmap(axis, matrix, title, cmap, norm=None, fmt=".2f"):
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="equal")
    axis.set_title(title)
    axis.set_xticks(range(matrix.shape[1]), labels=range(matrix.shape[1]))
    axis.set_yticks(range(matrix.shape[0]), labels=range(1, matrix.shape[0] + 1))
    axis.set_xlabel("old task j")
    axis.set_ylabel("checkpoint / transition k")
    for k in range(1, 10):
        for task in range(k):
            value = matrix[k - 1, task]
            if np.isfinite(value):
                axis.text(task, k - 1, format(value, fmt), ha="center", va="center", fontsize=7)
    return image


def correlation(rows, x_key, y_key):
    x = np.array([float(row[x_key]) for row in rows])
    y = np.array([float(row[y_key]) for row in rows])
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = read(args.metrics)
    fisher_stats = read(args.fisher_stats)
    overlaps = read(args.fisher_overlap)
    module_drift = read(args.drift_by_module)

    panels = [
        (
            "standard_forgetting_score",
            "Standard forgetting score",
            "Reds",
            None,
        ),
        (
            "cumulative_old_task_loss_increase",
            "Cumulative old-task loss increase",
            "RdBu_r",
            None,
        ),
        (
            "relative_fisher_cumulative_drift",
            "Fisher-weighted cumulative drift",
            "magma",
            None,
        ),
        (
            "relative_fisher_transition_drift",
            "Fisher-weighted transition drift",
            "magma",
            None,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    for axis, (key, title, cmap, _) in zip(axes.ravel(), panels):
        values = lower_matrix(metrics, key)
        if key == "cumulative_old_task_loss_increase":
            limit = max(0.1, float(np.nanpercentile(np.abs(values), 95)))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        elif key == "standard_forgetting_score":
            norm = plt.Normalize(0, 1)
        else:
            norm = plt.Normalize(0, float(np.nanpercentile(values, 95)))
        image = heatmap(axis, values, title, cmap, norm)
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("Direct forgetting and task-important parameter damage")
    fig.savefig(args.output / "four_metric_forgetting_dashboard.png", dpi=200)
    plt.close(fig)

    importance = np.full((10, len(MODULES)), np.nan)
    for row in fisher_stats:
        if row["module"] in MODULES:
            importance[int(row["task"]), MODULES.index(row["module"])] = math.log10(
                max(float(row["fisher_mean"]), 1e-30)
            )
    overlap = np.eye(10)
    for row in overlaps:
        if row["module"] == "all_action_head":
            a, b = int(row["task_a"]), int(row["task_b"])
            overlap[a, b] = overlap[b, a] = float(row["fisher_cosine"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    image = axes[0].imshow(importance, cmap="viridis", aspect="auto")
    axes[0].set_title("Diagonal Fisher / Hessian importance")
    axes[0].set_xticks(range(len(MODULES)), labels=[m.replace("_", "\n") for m in MODULES])
    axes[0].set_yticks(range(10), labels=range(10))
    axes[0].set_xlabel("action-head module")
    axes[0].set_ylabel("task")
    for task in range(10):
        for module in range(len(MODULES)):
            axes[0].text(module, task, f"{importance[task, module]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes[0], shrink=0.8, label="log10 mean diagonal Fisher")

    image = axes[1].imshow(overlap, vmin=0, vmax=1, cmap="magma")
    axes[1].set_title("Task-task Fisher importance overlap")
    axes[1].set_xticks(range(10), labels=range(10))
    axes[1].set_yticks(range(10), labels=range(10))
    axes[1].set_xlabel("task")
    axes[1].set_ylabel("task")
    for a in range(10):
        for b in range(10):
            axes[1].text(b, a, f"{overlap[a, b]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axes[1], shrink=0.8, label="cosine of diagonal Fisher vectors")
    fig.suptitle("Visualized diagonal empirical-Fisher Hessian approximation")
    fig.savefig(args.output / "fisher_hessian_approximation.png", dpi=200)
    plt.close(fig)

    correlations = {
        "fisher_cumulative_vs_forgetting": correlation(
            metrics, "relative_fisher_cumulative_drift", "standard_forgetting_score"
        ),
        "fisher_transition_vs_forgetting": correlation(
            metrics, "relative_fisher_transition_drift", "transition_success_drop"
        ),
        "fisher_cumulative_vs_loss": correlation(
            metrics,
            "relative_fisher_cumulative_drift",
            "cumulative_old_task_loss_increase",
        ),
        "fisher_transition_vs_loss": correlation(
            metrics,
            "relative_fisher_transition_drift",
            "transition_old_task_loss_increase",
        ),
    }
    (args.output / "fisher_behavior_correlations.json").write_text(
        json.dumps(correlations, indent=2) + "\n"
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, module in zip(axes.ravel(), MODULES):
        values = np.full((9, 9), np.nan)
        for row in module_drift:
            if row["module"] == module:
                values[int(row["checkpoint"]) - 1, int(row["reference_task"])] = float(
                    row["relative_fisher_cumulative_drift"]
                )
        image = heatmap(axis, values, module.replace("_", " "), "magma")
        fig.colorbar(image, ax=axis, shrink=0.7)
    fig.suptitle("Module-wise Fisher-weighted cumulative drift")
    fig.savefig(args.output / "fisher_cumulative_drift_by_module.png", dpi=200)
    plt.close(fig)
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
