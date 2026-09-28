#!/usr/bin/env python3
"""Build a staged catastrophic-forgetting analysis from existing fixed probes.

The analysis deliberately separates four levels:

1. behavior: rollout success drop;
2. gradient/parameter space: gradient direction and Fisher-weighted motion;
3. function space: an exact MSE decomposition of action-output change;
4. representation space: layer-wise linear-CKA drift.

For the fixed probe bank, the flow-matching MSE change is exactly

    delta_loss = 2 <residual_pre, delta_velocity> / N
                 + ||delta_velocity||^2 / N.

The first term is signed (helpful or harmful), while the second is an unsigned
movement cost.  The required loss and output-delta statistics were already
saved by ``analyze_forgetting_metrics.py``, so this decomposition introduces no
new stochastic forward passes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


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

MODULES = [
    "state_encoder",
    "action_encoder",
    "vl_projector",
    "future_tokens",
    "dit",
    "action_decoder",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    base = Path("outputs/analysis/libero_spatial/none")
    parser.add_argument(
        "--gradient-pairs",
        type=Path,
        default=base / "gradient_phase_change_seed42/pair_summary.csv",
    )
    parser.add_argument(
        "--gradient-modules",
        type=Path,
        default=base / "gradient_phase_change_seed42/gradient_phase_change.csv",
    )
    parser.add_argument(
        "--functional-metrics",
        type=Path,
        default=base / "forgetting_metrics_seed42/combined_forgetting_metrics.csv",
    )
    parser.add_argument(
        "--fisher-metrics",
        type=Path,
        default=base / "fisher_forgetting_seed42/four_forgetting_metrics.csv",
    )
    parser.add_argument(
        "--feature-metrics",
        type=Path,
        default=base / "phase_features_seed42/feature_transition_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "staged_forgetting_analysis_seed42",
    )
    return parser.parse_args()


def finite_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def transition_matrix(frame: pd.DataFrame, value: str) -> np.ndarray:
    matrix = np.full((9, 9), np.nan, dtype=np.float64)
    for row in frame[["transition", "old_task", value]].itertuples(index=False):
        matrix[int(row.transition) - 1, int(row.old_task)] = float(getattr(row, value))
    return matrix


def symmetric_limit(matrix: np.ndarray, floor: float = 0.05) -> float:
    values = np.abs(matrix[np.isfinite(matrix)])
    return max(floor, float(np.percentile(values, 95))) if len(values) else floor


def heatmap(axis, matrix, title, cmap, norm, fmt=".2f"):
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="equal")
    axis.set_title(title, fontsize=10)
    axis.set_xticks(range(9), labels=range(9))
    axis.set_yticks(range(9), labels=range(1, 10))
    axis.set_xlabel("old task j")
    axis.set_ylabel("transition k")
    for transition in range(1, 10):
        for old_task in range(transition):
            value = matrix[transition - 1, old_task]
            if np.isfinite(value):
                axis.text(
                    old_task,
                    transition - 1,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=6.3,
                )
    return image


def build_combined(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gradients = pd.read_csv(args.gradient_pairs)
    functional = pd.read_csv(args.functional_metrics)
    fisher = pd.read_csv(args.fisher_metrics).rename(columns={"checkpoint": "transition"})
    features = pd.read_csv(args.feature_metrics)

    keep_gradient = [
        "transition",
        "old_task",
        "global_old_drift_cosine",
        "global_cross_endpoint_cosine",
        "global_post_old_new_cosine",
        "global_old_norm_ratio",
        "global_old_log10_norm_ratio",
    ]
    keep_functional = [
        "transition",
        "old_task",
        "pre_old_loss",
        "post_old_loss",
        "old_loss_change",
        "velocity_output_delta_norm",
        "relative_velocity_output_drift",
        "velocity_output_cosine",
        "velocity_output_elements",
        "relative_parameter_displacement",
        "success_before",
        "success_after",
        "success_change",
        "forgetting_amount",
    ]
    keep_fisher = [
        "transition",
        "old_task",
        "standard_forgetting_score",
        "transition_success_drop",
        "relative_fisher_cumulative_drift",
        "relative_fisher_transition_drift",
    ]

    feature_wide = features.pivot(
        index=["transition", "old_task"],
        columns="feature",
        values="feature_drift_one_minus_cka",
    ).reset_index()
    feature_wide = feature_wide.rename(
        columns={feature: f"cka_drift_{feature}" for feature in FEATURES}
    )

    combined = (
        functional[keep_functional]
        .merge(gradients[keep_gradient], on=["transition", "old_task"], validate="one_to_one")
        .merge(fisher[keep_fisher], on=["transition", "old_task"], validate="one_to_one")
        .merge(feature_wide, on=["transition", "old_task"], validate="one_to_one")
        .sort_values(["transition", "old_task"])
        .reset_index(drop=True)
    )

    # Exact output-space decomposition for the masked flow-matching MSE.
    combined["output_quadratic_movement"] = (
        combined["velocity_output_delta_norm"] ** 2
        / combined["velocity_output_elements"]
    )
    combined["output_signed_interference"] = (
        combined["old_loss_change"] - combined["output_quadratic_movement"]
    )
    denominator = 2.0 * np.sqrt(
        combined["pre_old_loss"] * combined["output_quadratic_movement"]
    )
    combined["residual_output_change_cosine"] = (
        combined["output_signed_interference"] / denominator.replace(0, np.nan)
    ).clip(-1.0, 1.0)
    combined["decomposition_residual"] = combined["old_loss_change"] - (
        combined["output_signed_interference"]
        + combined["output_quadratic_movement"]
    )
    combined["at_risk_before_transition"] = combined["success_before"] > 0

    metric_labels = {
        "global_post_old_new_cosine": "Gradient: post old/new cosine",
        "global_old_drift_cosine": "Gradient: old direction retention",
        "global_old_log10_norm_ratio": "Gradient: log10 old norm ratio",
        "relative_parameter_displacement": "Actual parameter displacement",
        "relative_fisher_transition_drift": "Fisher-weighted transition drift",
        "output_signed_interference": "Output: signed interference",
        "output_quadratic_movement": "Output: unsigned movement",
        "old_loss_change": "Old-task flow-loss change",
        "cka_drift_vl_projector": "Feature: VL-projector CKA drift",
        "cka_drift_state_encoder": "Feature: state-encoder CKA drift",
        "cka_drift_action_encoder": "Feature: action-encoder CKA drift",
        "cka_drift_dit_early": "Feature: early-DiT CKA drift",
        "cka_drift_dit_middle": "Feature: middle-DiT CKA drift",
        "cka_drift_dit_late": "Feature: late-DiT CKA drift",
        "cka_drift_action_decoder": "Feature: action-decoder CKA drift",
    }
    correlations = []
    for subset_name, subset in (
        ("all_cells", combined),
        ("at_risk_only", combined[combined["at_risk_before_transition"]]),
    ):
        for metric, label in metric_labels.items():
            correlations.append(
                {
                    "subset": subset_name,
                    "metric": metric,
                    "label": label,
                    "n": len(subset),
                    "pearson_vs_transition_success_drop": finite_correlation(
                        subset[metric], subset["transition_success_drop"], "pearson"
                    ),
                    "spearman_vs_transition_success_drop": finite_correlation(
                        subset[metric], subset["transition_success_drop"], "spearman"
                    ),
                }
            )
    return combined, pd.DataFrame(correlations), features


def plot_overview(combined: pd.DataFrame, output: Path) -> None:
    panels = [
        (
            "success_change",
            "1. Rollout success change",
            "RdYlGn",
            TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        ),
        (
            "old_loss_change",
            "2. Old-task flow-loss change",
            "RdBu_r",
            None,
        ),
        (
            "global_post_old_new_cosine",
            "3. Raw-gradient old/new cosine",
            "coolwarm",
            TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        ),
        (
            "global_old_log10_norm_ratio",
            "4. log10 old-gradient norm ratio",
            "PuOr_r",
            TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2),
        ),
        (
            "relative_fisher_transition_drift",
            "5. Fisher-weighted parameter drift",
            "magma",
            None,
        ),
        (
            "output_signed_interference",
            "6. Signed output interference",
            "RdBu_r",
            None,
        ),
        (
            "output_quadratic_movement",
            "7. Unsigned output movement",
            "magma",
            None,
        ),
        (
            "cka_drift_vl_projector",
            "8. VL-projector CKA drift",
            "magma",
            None,
        ),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5), constrained_layout=True)
    for axis, (key, title, cmap, norm) in zip(axes.ravel(), panels):
        matrix = transition_matrix(combined, key)
        if norm is None:
            if key in {"old_loss_change", "output_signed_interference"}:
                limit = symmetric_limit(matrix)
                norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            else:
                upper = max(1e-8, float(np.nanpercentile(matrix, 95)))
                norm = plt.Normalize(0, upper)
        image = heatmap(axis, matrix, title, cmap, norm)
        fig.colorbar(image, ax=axis, shrink=0.72)
    fig.suptitle(
        "Catastrophic-forgetting analysis flow — LIBERO Spatial / none / seed 42",
        fontsize=15,
    )
    fig.savefig(output / "analysis_flow_overview_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_output_decomposition(combined: pd.DataFrame, output: Path) -> None:
    panels = [
        ("output_signed_interference", "Signed alignment term", "RdBu_r"),
        ("output_quadratic_movement", "Unsigned movement term", "magma"),
        ("old_loss_change", "Net flow-loss change", "RdBu_r"),
        ("success_change", "Rollout success change", "RdYlGn"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)
    for axis, (key, title, cmap) in zip(axes, panels):
        matrix = transition_matrix(combined, key)
        if key == "success_change":
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        elif key == "output_quadratic_movement":
            norm = plt.Normalize(0, max(1e-8, float(np.nanpercentile(matrix, 95))))
        else:
            limit = symmetric_limit(matrix)
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        image = heatmap(axis, matrix, title, cmap, norm)
        fig.colorbar(image, ax=axis, shrink=0.76)
    fig.suptitle(r"Exact function-space decomposition: $\Delta L=2\langle r,\Delta v\rangle+\|\Delta v\|^2$")
    fig.savefig(output / "signed_output_decomposition_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_gradient_modules(gradient_path: Path, output: Path) -> None:
    data = pd.read_csv(gradient_path)
    data = data[data["module"].isin(MODULES)]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    image = None
    for axis, module in zip(axes.ravel(), MODULES):
        frame = data[data["module"] == module]
        matrix = transition_matrix(frame, "post_old_new_cosine")
        image = heatmap(
            axis,
            matrix,
            module.replace("_", " "),
            "coolwarm",
            norm,
        )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="gradient cosine")
    fig.suptitle("Module-wise raw-gradient conflict after each phase")
    fig.savefig(output / "gradient_conflict_by_module_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_feature_modules(features: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    for axis, feature in zip(axes.ravel(), FEATURES):
        frame = features[features["feature"] == feature]
        matrix = transition_matrix(frame, "feature_drift_one_minus_cka")
        upper = max(1e-8, float(np.nanpercentile(matrix, 95)))
        image = heatmap(
            axis,
            matrix,
            feature.replace("_", " "),
            "magma",
            plt.Normalize(0, upper),
        )
        fig.colorbar(image, ax=axis, shrink=0.72)
    fig.suptitle("Representation-space drift by module (1 − linear CKA)")
    fig.savefig(output / "feature_drift_by_module_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_correlations(correlations: pd.DataFrame, output: Path) -> None:
    all_cells = correlations[correlations["subset"] == "all_cells"].copy()
    at_risk = correlations[correlations["subset"] == "at_risk_only"].copy()
    labels = all_cells["label"].tolist()
    y = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    axis.barh(
        y - 0.19,
        all_cells["pearson_vs_transition_success_drop"],
        height=0.36,
        label="all 45 cells",
    )
    axis.barh(
        y + 0.19,
        at_risk["pearson_vs_transition_success_drop"],
        height=0.36,
        label="13 cells with success_before > 0",
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlim(-1, 1)
    axis.set_yticks(y, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Pearson correlation with transition success drop (positive = more forgetting)")
    axis.set_title("Which signal tracks behavioral forgetting?")
    axis.legend()
    fig.savefig(output / "metric_vs_forgetting_correlations.png", dpi=220)
    plt.close(fig)


def write_summary(combined: pd.DataFrame, correlations: pd.DataFrame, output: Path) -> None:
    all_corr = correlations[correlations["subset"] == "all_cells"].set_index("metric")
    risk_corr = correlations[correlations["subset"] == "at_risk_only"].set_index("metric")
    max_residual = float(combined["decomposition_residual"].abs().max())
    lines = [
        "# Staged catastrophic-forgetting analysis",
        "",
        "## Design",
        "",
        "1. Confirm behavioral forgetting with rollout success change.",
        "2. Test module-wise raw-gradient direction conflict.",
        "3. Compare it with actual/Fisher-weighted parameter motion under AdamW.",
        "4. Move to function space (signed output interference) and representation space (CKA drift).",
        "",
        "All cells use transition `k` on the vertical axis and old task `j < k` on the horizontal axis.",
        "Fixed demonstrations, preprocessing, Gaussian noise, and flow timestep are shared across checkpoints.",
        "",
        "## Exact function-space identity",
        "",
        "`old_loss_change = output_signed_interference + output_quadratic_movement`",
        "",
        f"Maximum numerical decomposition residual: `{max_residual:.3e}`.",
        "",
        "## Pearson correlation with transition success drop",
        "",
        "| Signal | All 45 cells | At-risk cells only |",
        "|---|---:|---:|",
    ]
    ordered = [
        "global_post_old_new_cosine",
        "global_old_drift_cosine",
        "global_old_log10_norm_ratio",
        "relative_parameter_displacement",
        "relative_fisher_transition_drift",
        "output_signed_interference",
        "output_quadratic_movement",
        "old_loss_change",
        "cka_drift_vl_projector",
        "cka_drift_action_encoder",
        "cka_drift_dit_middle",
        "cka_drift_action_decoder",
    ]
    for metric in ordered:
        label = str(all_corr.loc[metric, "label"])
        all_value = float(all_corr.loc[metric, "pearson_vs_transition_success_drop"])
        risk_value = float(risk_corr.loc[metric, "pearson_vs_transition_success_drop"])
        lines.append(f"| {label} | {all_value:.3f} | {risk_value:.3f} |")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Gradient norm growth is a post-transition diagnostic, not necessarily a predictor or cause.",
        "- Fisher drift and unsigned output movement cannot distinguish damage from recovery because both discard direction.",
        "- Signed output interference is optimizer-independent and distinguishes helpful from harmful functional movement.",
        "- CKA association localizes representation change but is not causal; module restoration/freezing is required for causal confirmation.",
        "- This is one training seed; the 30 rollout trials reduce evaluation noise but do not replace multi-seed training.",
        "",
    ]
    (output / "RESULTS_AND_DESIGN.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined, correlations, features = build_combined(args)
    combined.to_csv(args.output / "analysis_flow_metrics.csv", index=False)
    correlations.to_csv(args.output / "metric_forgetting_correlations.csv", index=False)

    plot_overview(combined, args.output)
    plot_output_decomposition(combined, args.output)
    plot_gradient_modules(args.gradient_modules, args.output)
    plot_feature_modules(features, args.output)
    plot_correlations(correlations, args.output)
    write_summary(combined, correlations, args.output)

    metadata = {
        "num_transition_old_task_cells": len(combined),
        "num_at_risk_cells": int(combined["at_risk_before_transition"].sum()),
        "output_loss_decomposition_max_abs_residual": float(
            combined["decomposition_residual"].abs().max()
        ),
        "coordinate_system": "row=transition k (1..9), column=old task j (0..8), j<k",
        "new_forward_passes": 0,
        "analysis_scope": "LIBERO Spatial, none, training seed 42",
    }
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote staged analysis to {args.output}")


if __name__ == "__main__":
    main()
