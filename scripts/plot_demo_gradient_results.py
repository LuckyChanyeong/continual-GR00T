#!/usr/bin/env python3
"""Plot and summarize exact new-task versus old-task gradient alignment."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))

    expected = 45 * len(MODULES)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}")

    summary = []
    for module in MODULES:
        values = np.array([float(r["cosine"]) for r in rows if r["module"] == module])
        summary.append(
            {
                "module": module,
                "num_pairs": len(values),
                "mean_cosine": float(values.mean()),
                "median_cosine": float(np.median(values)),
                "min_cosine": float(values.min()),
                "max_cosine": float(values.max()),
                "negative_pairs": int((values < 0).sum()),
                "negative_fraction": float((values < 0).mean()),
            }
        )
    write_csv(args.output / "module_summary.csv", summary)

    conflicts = sorted(rows, key=lambda r: float(r["cosine"]))
    write_csv(args.output / "strongest_conflicts.csv", conflicts)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.ravel()
    image = None
    for axis, module in zip(axes, MODULES):
        matrix = np.full((9, 9), np.nan, dtype=np.float64)
        for row in rows:
            if row["module"] == module:
                matrix[int(row["incoming_task"]) - 1, int(row["old_task"])] = float(row["cosine"])
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(
            masked,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
            origin="upper",
            aspect="equal",
        )
        axis.set_title(module.replace("_", " "))
        axis.set_xticks(range(9), labels=range(9))
        axis.set_yticks(range(9), labels=range(1, 10))
        axis.set_xlabel("old task j")
        axis.set_ylabel("incoming task k")
        for incoming in range(1, 10):
            for old in range(incoming):
                value = matrix[incoming - 1, old]
                color = "white" if abs(value) >= 0.5 else "black"
                axis.text(old, incoming - 1, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    axes[-1].axis("off")
    assert image is not None
    fig.colorbar(image, ax=axes.tolist(), shrink=0.85, label="gradient cosine similarity")
    fig.suptitle("LIBERO Spatial: incoming-task vs old-task flow-gradient alignment (seed 42)")
    fig.savefig(args.output / "gradient_alignment_heatmaps.png", dpi=200)
    plt.close(fig)

    print(f"Wrote summaries and heatmap to {args.output}")


if __name__ == "__main__":
    main()
