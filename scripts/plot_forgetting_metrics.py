#!/usr/bin/env python3
"""Plot and summarize parameter/loss/output/rollout forgetting metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parameter-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def matrix(rows: list[dict[str, str]], key: str) -> np.ndarray:
    result = np.full((9, 9), np.nan)
    for row in rows:
        result[int(row["transition"]) - 1, int(row["old_task"])] = float(row[key])
    return result


def heatmap(axis, values, title, cmap, norm=None, fmt=".2f"):
    image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="equal")
    axis.set_title(title)
    axis.set_xticks(range(9), labels=range(9))
    axis.set_yticks(range(9), labels=range(1, 10))
    axis.set_xlabel("old task j")
    axis.set_ylabel("transition k")
    for transition in range(1, 10):
        for old_task in range(transition):
            value = values[transition - 1, old_task]
            if np.isfinite(value):
                axis.text(
                    old_task,
                    transition - 1,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    return image


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=float)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2
            start = end
        return result

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def correlation(rows, x_key, y_key):
    x = np.array([float(row[x_key]) for row in rows])
    y = np.array([float(row[y_key]) for row in rows])
    valid = np.isfinite(x) & np.isfinite(y)
    return {
        "n": int(valid.sum()),
        "pearson": float(np.corrcoef(x[valid], y[valid])[0, 1]),
        "spearman": rank_correlation(x[valid], y[valid]),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    parameter_rows = [
        row for row in read_rows(args.parameter_input) if row["module"] == "all_action_head"
    ]

    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    transitions = [int(row["transition"]) for row in parameter_rows]
    displacement = [float(row["relative_parameter_displacement"]) for row in parameter_rows]
    axes[0, 0].bar(transitions, displacement, color="#4C78A8")
    axes[0, 0].set_title("Actual action-head parameter displacement")
    axes[0, 0].set_xlabel("transition k (checkpoint k-1 -> k)")
    axes[0, 0].set_ylabel("||theta_k - theta_k-1|| / ||theta_k-1||")
    axes[0, 0].set_xticks(range(1, 10))
    axes[0, 0].grid(axis="y", alpha=0.25)

    loss_values = matrix(rows, "old_loss_change")
    loss_limit = max(0.1, float(np.nanpercentile(np.abs(loss_values), 95)))
    image = heatmap(
        axes[0, 1],
        loss_values,
        "Old-task flow-loss change",
        "RdBu_r",
        TwoSlopeNorm(vmin=-loss_limit, vcenter=0, vmax=loss_limit),
    )
    figure.colorbar(image, ax=axes[0, 1], shrink=0.75)

    drift_values = matrix(rows, "relative_velocity_output_drift")
    image = heatmap(
        axes[1, 0], drift_values, "Relative velocity-output drift", "magma", fmt=".2f"
    )
    figure.colorbar(image, ax=axes[1, 0], shrink=0.75)

    success_values = matrix(rows, "success_change")
    image = heatmap(
        axes[1, 1],
        success_values,
        "Rollout success change",
        "RdYlGn",
        TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
    )
    figure.colorbar(image, ax=axes[1, 1], shrink=0.75)
    figure.suptitle("Continual forgetting: optimizer result to behavior (seed 42)")
    figure.savefig(args.output / "forgetting_metrics_dashboard.png", dpi=200)
    plt.close(figure)

    correlations = {
        "loss_change_vs_forgetting": correlation(rows, "old_loss_change", "forgetting_amount"),
        "output_drift_vs_forgetting": correlation(
            rows, "relative_velocity_output_drift", "forgetting_amount"
        ),
        "output_drift_vs_loss_change": correlation(
            rows, "relative_velocity_output_drift", "old_loss_change"
        ),
    }
    transition_means = []
    for transition in range(1, 10):
        selected = [row for row in rows if int(row["transition"]) == transition]
        parameter = next(
            float(row["relative_parameter_displacement"])
            for row in parameter_rows
            if int(row["transition"]) == transition
        )
        transition_means.append(
            {
                "transition": transition,
                "relative_parameter_displacement": parameter,
                "mean_old_loss_change": float(
                    np.mean([float(row["old_loss_change"]) for row in selected])
                ),
                "mean_relative_velocity_output_drift": float(
                    np.mean([float(row["relative_velocity_output_drift"]) for row in selected])
                ),
                "mean_forgetting_amount": float(
                    np.mean([float(row["forgetting_amount"]) for row in selected])
                ),
            }
        )
    with (args.output / "correlations.json").open("w") as f:
        json.dump(correlations, f, indent=2)
    with (args.output / "transition_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(transition_means[0]))
        writer.writeheader()
        writer.writerows(transition_means)

    most_forgotten = sorted(rows, key=lambda row: float(row["forgetting_amount"]), reverse=True)[:10]
    lines = [
        "# Forgetting metrics summary",
        "",
        "Positive `forgetting_amount` means rollout success decreased. Parameter displacement is",
        "shared by every old task in a transition; loss/output/success are task-specific.",
        "",
        "## Pair-level correlations",
        "",
        "```json",
        json.dumps(correlations, indent=2),
        "```",
        "",
        "## Largest rollout decreases",
        "",
        "| transition | old task | success change | loss change | relative output drift |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in most_forgotten:
        lines.append(
            f"| {row['transition']} | {row['old_task']} | "
            f"{float(row['success_change']):+.3f} | {float(row['old_loss_change']):+.3f} | "
            f"{float(row['relative_velocity_output_drift']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Correlations describe association, not causality. Rollout success has a 30-trial",
            "resolution and many floor values, so inspect individual task pairs as well as the",
            "aggregate coefficients.",
        ]
    )
    (args.output / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote dashboard and summaries to {args.output}")


if __name__ == "__main__":
    main()
