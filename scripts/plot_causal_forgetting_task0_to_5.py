#!/usr/bin/env python3
"""Combine the first-six-task forgetting analyses into readable heatmaps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


MAX_TASK = 5
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
    root = Path("outputs/analysis/libero_spatial/none/causal_task0_to_5_seed42")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "figures")
    return parser.parse_args()


def matrix(frame: pd.DataFrame, key: str) -> np.ndarray:
    values = np.full((MAX_TASK, MAX_TASK), np.nan, dtype=np.float64)
    for row in frame[["transition", "old_task", key]].itertuples(index=False):
        transition = int(row.transition)
        old_task = int(row.old_task)
        if 1 <= transition <= MAX_TASK and 0 <= old_task < transition:
            values[transition - 1, old_task] = float(getattr(row, key))
    return values


def div_norm(values: np.ndarray, limit: float | None = None):
    if limit is None:
        finite = np.abs(values[np.isfinite(values)])
        limit = max(1e-12, float(finite.max())) if len(finite) else 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def seq_norm(values: np.ndarray, floor: float = 0.0):
    finite = values[np.isfinite(values)]
    upper = max(floor + 1e-12, float(finite.max())) if len(finite) else 1.0
    return plt.Normalize(vmin=floor, vmax=upper)


def text_color(image, value: float) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.56 else "white"


def draw(axis, values, title, cmap, norm, fmt=".3f"):
    image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="equal")
    axis.set_facecolor("#eeeeee")
    axis.set_title(title, fontsize=10.5, pad=8)
    axis.set_xticks(range(MAX_TASK), labels=range(MAX_TASK))
    axis.set_yticks(range(MAX_TASK), labels=range(1, MAX_TASK + 1))
    axis.set_xlabel("old task j")
    axis.set_ylabel("transition k")
    axis.set_xticks(np.arange(-0.5, MAX_TASK, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, MAX_TASK, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for transition in range(1, MAX_TASK + 1):
        for old_task in range(transition):
            value = values[transition - 1, old_task]
            if np.isfinite(value):
                axis.text(
                    old_task,
                    transition - 1,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="semibold",
                    color=text_color(image, value),
                )
    return image


def load_data(root: Path):
    functional = pd.read_csv(root / "functional/combined_forgetting_metrics.csv")
    updates = pd.read_csv(root / "module_update_projection/module_g_dot_delta.csv")
    swaps = pd.read_csv(root / "module_swap/module_swap_recovery.csv")
    features = pd.read_csv(root / "features/feature_transition_metrics.csv")
    fisher = pd.read_csv(root / "fisher/four_forgetting_metrics.csv").rename(
        columns={"checkpoint": "transition"}
    )
    fisher_modules = pd.read_csv(root / "fisher/fisher_module_statistics.csv")
    fisher_overlap = pd.read_csv(root / "fisher/fisher_task_overlap.csv")

    functional["output_quadratic_movement"] = (
        functional["velocity_output_delta_norm"] ** 2
        / functional["velocity_output_elements"]
    )
    functional["output_signed_alignment"] = (
        functional["old_loss_change"] - functional["output_quadratic_movement"]
    )
    functional["alignment_compensation_ratio"] = (
        -functional["output_signed_alignment"]
        / functional["output_quadratic_movement"]
    )

    global_updates = updates[updates["module"] == "all_action_head"][
        ["transition", "old_task", "g_dot_delta", "gradient_delta_cosine"]
    ].rename(
        columns={
            "g_dot_delta": "global_g_dot_delta",
            "gradient_delta_cosine": "global_gradient_delta_cosine",
        }
    )
    feature_wide = features.pivot(
        index=["transition", "old_task"],
        columns="feature",
        values="feature_drift_one_minus_cka",
    ).reset_index()
    feature_wide = feature_wide.rename(
        columns={name: f"cka_drift_{name}" for name in FEATURES}
    )
    cells = (
        functional.merge(global_updates, on=["transition", "old_task"], validate="one_to_one")
        .merge(
            fisher[
                [
                    "transition",
                    "old_task",
                    "relative_fisher_transition_drift",
                    "relative_fisher_cumulative_drift",
                ]
            ],
            on=["transition", "old_task"],
            validate="one_to_one",
        )
        .merge(feature_wide, on=["transition", "old_task"], validate="one_to_one")
    )
    return cells, updates, swaps, features, fisher_modules, fisher_overlap


def plot_overview(cells: pd.DataFrame, swaps: pd.DataFrame, output: Path):
    dit_swap = swaps[
        (swaps["source_mode"] == "transition") & (swaps["module"] == "dit")
    ][["transition", "old_task", "absolute_loss_recovery"]]
    overview = cells.merge(dit_swap, on=["transition", "old_task"], validate="one_to_one")
    panels = [
        ("success_change", "Rollout success change", "RdYlGn", "div", 1.0),
        ("old_loss_change", "Net old-task loss change", "RdBu_r", "div", None),
        ("output_signed_alignment", "Signed residual-alignment", "RdBu_r", "div", None),
        ("output_quadratic_movement", "Unsigned output movement", "viridis", "seq", None),
        ("global_g_dot_delta", r"Actual update: $g^T\Delta\theta$", "RdBu_r", "div", None),
        ("absolute_loss_recovery", "DiT transition-swap recovery", "RdYlGn", "div", None),
        ("cka_drift_vl_projector", "VL-projector drift (1−CKA)", "viridis", "seq", None),
        ("relative_fisher_transition_drift", "Fisher-weighted drift", "cividis", "seq", None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    for axis, (key, title, cmap, kind, fixed) in zip(axes.ravel(), panels):
        values = matrix(overview, key)
        norm = div_norm(values, fixed) if kind == "div" else seq_norm(values)
        image = draw(axis, values, title, cmap, norm)
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle("Causal forgetting analysis — LIBERO Spatial tasks 0–5", fontsize=16)
    fig.savefig(output / "00_integrated_overview.png", dpi=240)
    plt.close(fig)


def plot_decomposition(cells: pd.DataFrame, output: Path):
    panels = [
        ("output_signed_alignment", "Signed residual alignment", "RdBu_r", "div"),
        ("output_quadratic_movement", "Unsigned movement cost", "viridis", "seq"),
        ("alignment_compensation_ratio", "Alignment compensation ratio", "RdYlGn", "ratio"),
        ("old_loss_change", "Net flow-loss change", "RdBu_r", "div"),
        ("success_change", "Rollout success change", "RdYlGn", "success"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(23, 5), constrained_layout=True)
    for axis, (key, title, cmap, kind) in zip(axes, panels):
        values = matrix(cells, key)
        if kind == "seq":
            norm = seq_norm(values)
        elif kind == "ratio":
            finite = values[np.isfinite(values)]
            lower = min(-1.0, float(finite.min()))
            upper = max(2.0, float(finite.max()))
            norm = TwoSlopeNorm(vmin=lower, vcenter=1.0, vmax=upper)
        elif kind == "success":
            norm = div_norm(values, 1.0)
        else:
            norm = div_norm(values)
        image = draw(axis, values, title, cmap, norm)
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle(
        r"Signed action-output loss decomposition: $\Delta L=2\langle r,\Delta v\rangle+\|\Delta v\|^2$",
        fontsize=15,
    )
    fig.savefig(output / "01_signed_action_output_loss_decomposition.png", dpi=240)
    plt.close(fig)


def plot_module_grid(frame, modules, key, title, cmap, output_path, colorbar_label="positive = predicted loss increase"):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.5), constrained_layout=True)
    for axis, module in zip(axes.ravel(), modules):
        selected = frame[frame["module"] == module]
        values = matrix(selected, key)
        norm = div_norm(values)
        image = draw(axis, values, module.replace("_", " "), cmap, norm)
        label = colorbar_label
        fig.colorbar(image, ax=axis, shrink=0.78, label=label)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_updates(updates: pd.DataFrame, output: Path):
    plot_module_grid(
        updates,
        MODULES,
        "g_dot_delta",
        r"Module-wise actual-update projection $g_j(\theta_{k-1})^T(\theta_k-\theta_{k-1})$",
        "RdBu_r",
        output / "02_modulewise_g_dot_delta.png",
    )


def plot_swaps(swaps: pd.DataFrame, output: Path):
    for mode, filename, explanation in (
        (
            "transition",
            "03a_transition_module_swap_recovery.png",
            "restore module from checkpoint k−1",
        ),
        (
            "acquisition",
            "03b_acquisition_module_swap_recovery.png",
            "restore module from old task acquisition checkpoint j",
        ),
    ):
        plot_module_grid(
            swaps[swaps["source_mode"] == mode],
            MODULES,
            "absolute_loss_recovery",
            f"Module-swap old-task loss recovery — {explanation}",
            "RdYlGn",
            output / filename,
            colorbar_label="positive = lower old-task loss",
        )


def plot_features(features: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), constrained_layout=True)
    for axis, feature in zip(axes.ravel(), FEATURES):
        selected = features[features["feature"] == feature]
        values = matrix(selected, "feature_drift_one_minus_cka")
        image = draw(
            axis,
            values,
            feature.replace("_", " "),
            "viridis",
            seq_norm(values),
        )
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle("Feature representation drift (1 − linear CKA)", fontsize=15)
    fig.savefig(output / "04_feature_cka_drift.png", dpi=240)
    plt.close(fig)


def plot_fisher(cells, stats, overlap_rows, root: Path, output: Path):
    modules = MODULES
    importance = np.full((MAX_TASK + 1, len(modules)), np.nan)
    for row in stats.itertuples(index=False):
        if row.module in modules:
            importance[int(row.task), modules.index(row.module)] = math.log10(
                max(float(row.fisher_mean), 1e-30)
            )
    overlap = np.eye(MAX_TASK + 1)
    for row in overlap_rows.itertuples(index=False):
        if row.module == "all_action_head":
            overlap[int(row.task_a), int(row.task_b)] = float(row.fisher_cosine)
            overlap[int(row.task_b), int(row.task_a)] = float(row.fisher_cosine)

    drift = matrix(cells, "relative_fisher_transition_drift")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    image = draw(axes[0], drift, "Fisher-weighted transition drift", "cividis", seq_norm(drift))
    fig.colorbar(image, ax=axes[0], shrink=0.8)

    image = axes[1].imshow(importance, cmap="viridis", aspect="auto")
    axes[1].set_title("log10 mean diagonal Fisher")
    axes[1].set_xticks(range(len(modules)), labels=[m.replace("_", "\n") for m in modules])
    axes[1].set_yticks(range(MAX_TASK + 1), labels=range(MAX_TASK + 1))
    axes[1].set_xlabel("module")
    axes[1].set_ylabel("task")
    for task in range(MAX_TASK + 1):
        for module in range(len(modules)):
            value = importance[task, module]
            axes[1].text(module, task, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], shrink=0.8)

    image = axes[2].imshow(overlap, cmap="magma", vmin=0, vmax=1, aspect="equal")
    axes[2].set_title("Task-task Fisher overlap")
    axes[2].set_xticks(range(MAX_TASK + 1), labels=range(MAX_TASK + 1))
    axes[2].set_yticks(range(MAX_TASK + 1), labels=range(MAX_TASK + 1))
    axes[2].set_xlabel("task")
    axes[2].set_ylabel("task")
    for left in range(MAX_TASK + 1):
        for right in range(MAX_TASK + 1):
            value = overlap[left, right]
            color = "white" if value < 0.55 else "black"
            axes[2].text(right, left, f"{value:.2f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(image, ax=axes[2], shrink=0.8)
    fig.suptitle("Diagonal empirical Fisher as a curvature auxiliary", fontsize=15)
    fig.savefig(output / "05_fisher_curvature_auxiliary.png", dpi=240)
    plt.close(fig)

    module_drift = pd.read_csv(root / "fisher/fisher_weighted_drift_by_module.csv")
    module_drift = module_drift.rename(columns={"checkpoint": "transition", "reference_task": "old_task"})
    plot_module_grid(
        module_drift,
        MODULES,
        "relative_fisher_transition_drift",
        "Module-wise Fisher-weighted transition drift",
        "RdBu_r",
        output / "05b_fisher_drift_by_module.png",
        colorbar_label="non-negative curvature-weighted drift",
    )


def write_summary(cells, updates, swaps, output: Path):
    global_updates = updates[updates["module"] == "all_action_head"]
    merged = cells.merge(
        global_updates[["transition", "old_task", "g_dot_delta"]],
        on=["transition", "old_task"],
        suffixes=("", "_check"),
    )
    transition_swap = swaps[swaps["source_mode"] == "transition"]
    acquisition_swap = swaps[swaps["source_mode"] == "acquisition"]
    transition_means = transition_swap.groupby("module")["absolute_loss_recovery"].mean().sort_values(ascending=False)
    acquisition_means = acquisition_swap.groupby("module")["absolute_loss_recovery"].mean().sort_values(ascending=False)

    correlations = {
        "signed_alignment_vs_loss_change": float(cells["output_signed_alignment"].corr(cells["old_loss_change"])),
        "quadratic_movement_vs_loss_change": float(cells["output_quadratic_movement"].corr(cells["old_loss_change"])),
        "g_dot_delta_vs_loss_change": float(merged["g_dot_delta"].corr(merged["old_loss_change"])),
        "vl_projector_cka_drift_vs_loss_change": float(cells["cka_drift_vl_projector"].corr(cells["old_loss_change"])),
        "fisher_transition_drift_vs_loss_change": float(cells["relative_fisher_transition_drift"].corr(cells["old_loss_change"])),
    }
    (output / "correlations.json").write_text(json.dumps(correlations, indent=2) + "\n")

    lines = [
        "# Task 0–5 causal forgetting analysis",
        "",
        "## Experimental order",
        "",
        "1. Signed action-output loss decomposition on identical fixed probes.",
        "2. Module-wise `g^T delta_theta` using the actual AdamW checkpoint displacement.",
        "3. Module-swap recovery from transition and task-acquisition checkpoints.",
        "4. Layer-wise feature drift using linear CKA.",
        "5. Sample-wise diagonal empirical Fisher as a curvature auxiliary.",
        "",
        "Each task uses 80 deterministic probes. All transition heatmaps use row `k=1..5` and column `j<k`.",
        "",
        "## Main numerical observations",
        "",
        f"- `g^T delta_theta` vs old-task loss change: `{correlations['g_dot_delta_vs_loss_change']:.3f}`.",
        f"- Signed output alignment vs old-task loss change: `{correlations['signed_alignment_vs_loss_change']:.3f}`.",
        f"- VL-projector CKA drift vs old-task loss change: `{correlations['vl_projector_cka_drift_vs_loss_change']:.3f}`.",
        f"- Fisher transition drift vs old-task loss change: `{correlations['fisher_transition_drift_vs_loss_change']:.3f}`.",
        f"- Mean transition-swap recovery is largest for `{transition_means.index[0]}`: `{transition_means.iloc[0]:.4f}` loss.",
        f"- Mean acquisition-swap recovery is largest for `{acquisition_means.index[0]}`: `{acquisition_means.iloc[0]:.4f}` loss.",
        "",
        "## Interpretation",
        "",
        "- Positive `g^T delta_theta` is a first-order harmful update; negative is first-order helpful. It can miss finite-step curvature/nonlinearity.",
        "- Positive module-swap recovery means restoring that module reduces old-task loss. Modules are not additive because the head is co-adapted.",
        "- CKA localizes representation restructuring but does not itself prove causality.",
        "- Fisher is retained only as a non-negative sensitivity/curvature measure; it cannot encode whether movement helps or hurts.",
        "- Fifteen transition-task cells from one training seed are descriptive, not a population-level significance test.",
        "",
    ]
    (output / "RESULTS_SUMMARY.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cells, updates, swaps, features, fisher_stats, fisher_overlap = load_data(args.root)
    cells.to_csv(args.output / "combined_cell_metrics.csv", index=False)
    plot_overview(cells, swaps, args.output)
    plot_decomposition(cells, args.output)
    plot_updates(updates, args.output)
    plot_swaps(swaps, args.output)
    plot_features(features, args.output)
    plot_fisher(cells, fisher_stats, fisher_overlap, args.root, args.output)
    write_summary(cells, updates, swaps, args.output)
    print(f"Wrote causal first-six-task heatmaps to {args.output}")


if __name__ == "__main__":
    main()
