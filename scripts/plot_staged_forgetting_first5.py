#!/usr/bin/env python3
"""Render readable heatmaps for the first five continual tasks (task 0..4)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


MAX_TASK = 4
MODULES = [
    "state_encoder",
    "action_encoder",
    "vl_projector",
    "future_tokens",
    "dit",
    "action_decoder",
]
FEATURES = [
    "backbone",
    "vl_projector",
    "state_encoder",
    "action_encoder",
    "dit_early",
    "dit_middle",
    "dit_late",
    "action_decoder",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    base = Path("outputs/analysis/libero_spatial/none")
    parser.add_argument(
        "--combined",
        type=Path,
        default=base / "staged_forgetting_analysis_seed42/analysis_flow_metrics.csv",
    )
    parser.add_argument(
        "--gradient-modules",
        type=Path,
        default=base / "gradient_phase_change_seed42/gradient_phase_change.csv",
    )
    parser.add_argument(
        "--feature-metrics",
        type=Path,
        default=base / "phase_features_seed42/feature_transition_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "staged_forgetting_analysis_seed42_task0_to_4",
    )
    return parser.parse_args()


def first_five(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["transition"] <= MAX_TASK) & (frame["old_task"] < MAX_TASK)].copy()


def make_matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    matrix = np.full((MAX_TASK, MAX_TASK), np.nan, dtype=np.float64)
    for row in frame[["transition", "old_task", value]].itertuples(index=False):
        transition = int(row.transition)
        old_task = int(row.old_task)
        if 1 <= transition <= MAX_TASK and old_task < transition:
            matrix[transition - 1, old_task] = float(getattr(row, value))
    return matrix


def sequential_norm(matrix: np.ndarray):
    values = matrix[np.isfinite(matrix)]
    upper = max(1e-9, float(values.max())) if len(values) else 1.0
    return plt.Normalize(vmin=0.0, vmax=upper)


def diverging_norm(matrix: np.ndarray, fixed_limit: float | None = None):
    if fixed_limit is None:
        values = np.abs(matrix[np.isfinite(matrix)])
        fixed_limit = max(1e-9, float(values.max())) if len(values) else 1.0
    return TwoSlopeNorm(vmin=-fixed_limit, vcenter=0.0, vmax=fixed_limit)


def readable_text_color(image, value: float) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    # Relative perceived luminance.  A slightly high threshold keeps text
    # readable on saturated yellow/orange cells as well as dark blue cells.
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.56 else "white"


def draw_heatmap(axis, matrix, title, cmap, norm, fmt=".2f"):
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="equal")
    axis.set_facecolor("#f2f2f2")
    axis.set_title(title, fontsize=11, pad=8)
    axis.set_xticks(range(MAX_TASK), labels=range(MAX_TASK))
    axis.set_yticks(range(MAX_TASK), labels=range(1, MAX_TASK + 1))
    axis.set_xlabel("old task j")
    axis.set_ylabel("transition k")
    axis.set_xticks(np.arange(-0.5, MAX_TASK, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, MAX_TASK, 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for transition in range(1, MAX_TASK + 1):
        for old_task in range(transition):
            value = matrix[transition - 1, old_task]
            if np.isfinite(value):
                axis.text(
                    old_task,
                    transition - 1,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="semibold",
                    color=readable_text_color(image, value),
                )
    return image


def plot_overview(data: pd.DataFrame, output: Path) -> None:
    panels = [
        ("success_change", "1. Rollout success change", "RdYlGn", "div", 1.0),
        ("old_loss_change", "2. Old-task flow-loss change", "RdBu_r", "div", None),
        ("global_post_old_new_cosine", "3. Raw-gradient old/new cosine", "coolwarm", "div", 1.0),
        ("global_old_log10_norm_ratio", "4. log10 old-gradient norm ratio", "PuOr_r", "div", None),
        ("relative_fisher_transition_drift", "5. Fisher-weighted parameter drift", "cividis", "seq", None),
        ("output_signed_interference", "6. Signed residual-alignment term", "RdBu_r", "div", None),
        ("output_quadratic_movement", "7. Unsigned output movement", "viridis", "seq", None),
        ("cka_drift_vl_projector", "8. VL-projector CKA drift", "viridis", "seq", None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.2), constrained_layout=True)
    for axis, (key, title, cmap, kind, limit) in zip(axes.ravel(), panels):
        matrix = make_matrix(data, key)
        norm = diverging_norm(matrix, limit) if kind == "div" else sequential_norm(matrix)
        image = draw_heatmap(axis, matrix, title, cmap, norm)
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle(
        "Staged catastrophic-forgetting analysis — tasks 0–4 (seed 42)",
        fontsize=16,
    )
    fig.savefig(output / "task0_to_4_analysis_flow_heatmaps.png", dpi=240)
    plt.close(fig)


def plot_decomposition(data: pd.DataFrame, output: Path) -> None:
    data = data.copy()
    data["alignment_compensation_ratio"] = (
        -data["output_signed_interference"] / data["output_quadratic_movement"]
    )
    panels = [
        ("output_signed_interference", "Signed residual alignment", "RdBu_r", "div"),
        ("output_quadratic_movement", "Unsigned movement cost", "viridis", "seq"),
        ("alignment_compensation_ratio", "Alignment compensation ratio", "RdYlGn", "ratio"),
        ("old_loss_change", "Net flow-loss change", "RdBu_r", "div"),
        ("success_change", "Rollout success change", "RdYlGn", "success"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.9), constrained_layout=True)
    for axis, (key, title, cmap, kind) in zip(axes, panels):
        matrix = make_matrix(data, key)
        if kind == "seq":
            norm = sequential_norm(matrix)
        elif kind == "ratio":
            # Ratio > 1 means beneficial alignment outweighs movement cost.
            limit = max(1.0, float(np.nanmax(np.abs(matrix))))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=1.0, vmax=limit + 1e-6)
        elif kind == "success":
            norm = diverging_norm(matrix, 1.0)
        else:
            norm = diverging_norm(matrix)
        image = draw_heatmap(axis, matrix, title, cmap, norm)
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle(
        r"Function-space decomposition: $\Delta L=2\langle r,\Delta v\rangle+\|\Delta v\|^2$",
        fontsize=15,
    )
    fig.savefig(output / "task0_to_4_signed_output_decomposition.png", dpi=240)
    plt.close(fig)


def plot_gradient_modules(path: Path, output: Path) -> None:
    data = first_five(pd.read_csv(path))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.6), constrained_layout=True)
    norm = diverging_norm(np.array([-1.0, 1.0]), 1.0)
    image = None
    for axis, module in zip(axes.ravel(), MODULES):
        matrix = make_matrix(data[data["module"] == module], "post_old_new_cosine")
        image = draw_heatmap(axis, matrix, module.replace("_", " "), "coolwarm", norm)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="gradient cosine")
    fig.suptitle("Module-wise raw-gradient conflict — tasks 0–4", fontsize=15)
    fig.savefig(output / "task0_to_4_gradient_conflict_by_module.png", dpi=240)
    plt.close(fig)


def plot_feature_modules(path: Path, output: Path) -> None:
    data = first_five(pd.read_csv(path))
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), constrained_layout=True)
    for axis, feature in zip(axes.ravel(), FEATURES):
        matrix = make_matrix(
            data[data["feature"] == feature], "feature_drift_one_minus_cka"
        )
        image = draw_heatmap(
            axis,
            matrix,
            feature.replace("_", " "),
            "viridis",
            sequential_norm(matrix),
        )
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle("Representation drift (1 − linear CKA) — tasks 0–4", fontsize=15)
    fig.savefig(output / "task0_to_4_feature_drift_by_module.png", dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined = first_five(pd.read_csv(args.combined))
    combined["alignment_compensation_ratio"] = (
        -combined["output_signed_interference"]
        / combined["output_quadratic_movement"]
    )
    combined.to_csv(args.output / "task0_to_4_analysis_metrics.csv", index=False)
    plot_overview(combined, args.output)
    plot_decomposition(combined, args.output)
    plot_gradient_modules(args.gradient_modules, args.output)
    plot_feature_modules(args.feature_metrics, args.output)
    print(f"Wrote first-five-task heatmaps to {args.output}")


if __name__ == "__main__":
    main()
