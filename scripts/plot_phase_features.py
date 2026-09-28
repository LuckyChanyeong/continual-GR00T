#!/usr/bin/env python3
"""Visualize feature retention/drift and relate it to behavioral forgetting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-input", type=Path, required=True)
    parser.add_argument("--retention-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def transition_matrix(rows, feature, key):
    values = np.full((9, 9), np.nan)
    for row in rows:
        if row["feature"] == feature:
            values[int(row["transition"]) - 1, int(row["old_task"])] = float(row[key])
    return values


def retention_matrix(rows, feature, key):
    values = np.full((10, 10), np.nan)
    for row in rows:
        if row["feature"] == feature:
            values[int(row["checkpoint"]), int(row["task"])] = float(row[key])
    return values


def draw(axis, matrix, title, vmin, vmax, cmap, x_label, y_labels):
    image = axis.imshow(np.ma.masked_invalid(matrix), vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_title(title)
    axis.set_xticks(range(matrix.shape[1]), labels=range(matrix.shape[1]))
    axis.set_yticks(range(matrix.shape[0]), labels=y_labels)
    axis.set_xlabel(x_label)
    return image


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    transitions = read(args.transition_input)
    retention = read(args.retention_input)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    image = None
    for axis, feature in zip(axes.ravel(), FEATURES):
        image = draw(
            axis,
            transition_matrix(transitions, feature, "feature_drift_one_minus_cka"),
            feature.replace("_", " "),
            0,
            1,
            "magma",
            "old task j",
            range(1, 10),
        )
        axis.set_ylabel("transition k")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="feature drift = 1 - CKA")
    fig.suptitle("Consecutive-phase feature drift on fixed old-task probes")
    fig.savefig(args.output / "feature_transition_drift_by_layer.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for axis, feature in zip(axes.ravel(), FEATURES):
        image = draw(
            axis,
            retention_matrix(retention, feature, "retention_cka"),
            feature.replace("_", " "),
            0,
            1,
            "viridis",
            "task j",
            range(10),
        )
        axis.set_ylabel("checkpoint k")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="CKA vs acquisition checkpoint")
    fig.suptitle("Feature retention relative to each task's acquisition checkpoint")
    fig.savefig(args.output / "feature_retention_by_layer.png", dpi=200)
    plt.close(fig)

    correlation_rows = []
    for feature in FEATURES:
        selected = [row for row in transitions if row["feature"] == feature]
        drift = [float(row["feature_drift_one_minus_cka"]) for row in selected]
        displacement = [float(row["consecutive_normalized_displacement"]) for row in selected]
        forgetting = [float(row["forgetting_amount"]) for row in selected]
        loss = [float(row["old_loss_change"]) for row in selected]
        correlation_rows.append(
            {
                "feature": feature,
                "cka_drift_vs_forgetting_pearson": corr(drift, forgetting),
                "cka_drift_vs_loss_change_pearson": corr(drift, loss),
                "displacement_vs_forgetting_pearson": corr(displacement, forgetting),
                "mean_cka_drift": float(np.mean(drift)),
            }
        )
    with (args.output / "feature_behavior_correlations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(correlation_rows[0]))
        writer.writeheader()
        writer.writerows(correlation_rows)

    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(FEATURES))
    width = 0.36
    axis.bar(
        x - width / 2,
        [row["cka_drift_vs_forgetting_pearson"] for row in correlation_rows],
        width,
        label="vs rollout forgetting",
    )
    axis.bar(
        x + width / 2,
        [row["cka_drift_vs_loss_change_pearson"] for row in correlation_rows],
        width,
        label="vs old-task loss increase",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels=[name.replace("_", "\n") for name in FEATURES])
    axis.set_ylabel("Pearson correlation")
    axis.set_title("Where feature drift is associated with forgetting")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(args.output / "feature_behavior_correlations.png", dpi=200)
    plt.close(fig)

    print(json.dumps(correlation_rows, indent=2))


if __name__ == "__main__":
    main()
