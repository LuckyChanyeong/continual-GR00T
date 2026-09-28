#!/usr/bin/env python3
"""Plot and summarize old-gradient versus actual checkpoint displacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/analysis/libero_spatial/none/update_alignment_v1"),
    )
    return parser.parse_args()


def matrix(frame: pd.DataFrame, key: str) -> np.ndarray:
    result = np.full((5, 5), np.nan)
    for row in frame[["transition", "old_task", key]].itertuples(index=False):
        result[int(row.transition) - 1, int(row.old_task)] = float(getattr(row, key))
    return result


def norm_for(values: np.ndarray, fixed: float | None = None):
    limit = fixed
    if limit is None:
        finite = np.abs(values[np.isfinite(values)])
        limit = float(finite.max()) if len(finite) else 1.0
    limit = max(limit, 1e-12)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def style_axis(axis):
    axis.set_facecolor("#e5e5e5")
    axis.set_xticks(range(5), labels=[f"Task {i}" for i in range(5)])
    axis.set_yticks(range(5), labels=[f"C{i}→C{i+1}" for i in range(5)])
    axis.tick_params(axis="x", rotation=35)
    axis.set_xlabel("Old task")
    axis.set_ylabel("Checkpoint transition")
    axis.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)


def annotate(axis, image, values, fmt):
    for transition in range(1, 6):
        for old_task in range(transition):
            value = values[transition - 1, old_task]
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                old_task,
                transition - 1,
                format(value, fmt),
                ha="center",
                va="center",
                color="black" if luminance > 0.58 else "white",
                fontsize=9,
                fontweight="bold",
            )


def draw(axis, values, title, fmt, fixed=None):
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap="RdBu_r",
        norm=norm_for(values, fixed),
        aspect="equal",
    )
    style_axis(axis)
    axis.set_title(title, fontsize=11)
    annotate(axis, image, values, fmt)
    plt.colorbar(image, ax=axis, shrink=0.82)


def main():
    args = parse_args()
    frame = pd.read_csv(args.root / "update_alignment.csv")
    frame["higher_order_remainder"] = (
        frame["actual_loss_change_mean"] - frame["g_dot_delta"]
    )
    frame.to_csv(args.root / "update_alignment_with_remainder.csv", index=False)
    figures = args.root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    draw(
        axes[0, 0],
        matrix(frame, "optimization_update_cosine"),
        "Old-task optimization vs actual AdamW displacement\nnegative = directional conflict",
        "+.3f",
        1.0,
    )
    draw(
        axes[0, 1],
        matrix(frame, "g_dot_delta"),
        "First-order predicted loss change: gᵀΔθ\npositive = predicted damage",
        "+.3f",
    )
    draw(
        axes[1, 0],
        matrix(frame, "actual_loss_change_mean"),
        "Measured fixed-probe flow-loss change\npositive = offline forgetting",
        "+.3f",
    )
    draw(
        axes[1, 1],
        matrix(frame, "success_change") * 100,
        "Rollout success-rate change (percentage points)\nnegative = behavioral forgetting",
        "+.1f",
        100.0,
    )
    fig.suptitle("Old-task gradient vs actual cumulative AdamW displacement", fontsize=16)
    fig.savefig(figures / "00_update_alignment_overview.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    draw(axes[0], matrix(frame, "g_dot_delta"), "First-order term: gᵀΔθ", "+.3f")
    draw(
        axes[1],
        matrix(frame, "actual_loss_change_mean"),
        "Measured loss change: ΔL",
        "+.3f",
    )
    remainder_values = matrix(frame, "higher_order_remainder")
    remainder_image = axes[2].imshow(
        np.ma.masked_invalid(remainder_values),
        cmap="viridis",
        norm=plt.Normalize(vmin=0.0, vmax=float(np.nanmax(remainder_values))),
        aspect="equal",
    )
    style_axis(axes[2])
    axes[2].set_title("Higher-order remainder: ΔL − gᵀΔθ", fontsize=11)
    annotate(axes[2], remainder_image, remainder_values, ".3f")
    plt.colorbar(remainder_image, ax=axes[2], shrink=0.82)
    fig.suptitle("Why the endpoint first-order term misses forgetting", fontsize=15)
    fig.savefig(figures / "01_first_order_and_remainder.png", dpi=240)
    plt.close(fig)

    predicted_harm = frame["g_dot_delta"] > 0
    measured_harm = frame["actual_loss_change_mean"] > 0
    sign_agreement = float((predicted_harm == measured_harm).mean())
    correlations = {
        "num_cells": int(len(frame)),
        "predicted_vs_actual_loss_pearson": float(
            frame["g_dot_delta"].corr(frame["actual_loss_change_mean"])
        ),
        "predicted_vs_actual_loss_spearman": float(
            frame["g_dot_delta"].corr(frame["actual_loss_change_mean"], method="spearman")
        ),
        "predicted_actual_loss_sign_agreement": sign_agreement,
        "negative_optimization_update_cosine_cells": int(
            (frame["optimization_update_cosine"] < 0).sum()
        ),
        "positive_actual_loss_change_cells": int(measured_harm.sum()),
        "max_absolute_optimization_update_cosine": float(
            frame["optimization_update_cosine"].abs().max()
        ),
        "median_absolute_optimization_update_cosine": float(
            frame["optimization_update_cosine"].abs().median()
        ),
        "positive_higher_order_remainder_cells": int(
            (frame["higher_order_remainder"] > 0).sum()
        ),
        "minimum_higher_order_remainder": float(frame["higher_order_remainder"].min()),
        "maximum_higher_order_remainder": float(frame["higher_order_remainder"].max()),
    }
    at_risk = frame[frame["success_before"] > 0]
    correlations["num_at_risk_cells"] = int(len(at_risk))
    correlations["predicted_damage_vs_success_change_at_risk_pearson"] = float(
        at_risk["g_dot_delta"].corr(at_risk["success_change"])
    )
    forgetting = frame[frame["success_change"] < 0]
    correlations["behavioral_forgetting_cells"] = int(len(forgetting))
    correlations["behavioral_forgetting_cells_with_positive_g_dot_delta"] = int(
        (forgetting["g_dot_delta"] > 0).sum()
    )
    centered = frame.copy()
    centered["centered_g_dot_delta"] = centered["g_dot_delta"] - centered.groupby(
        "transition"
    )["g_dot_delta"].transform("mean")
    centered["centered_loss_change"] = centered["actual_loss_change_mean"] - centered.groupby(
        "transition"
    )["actual_loss_change_mean"].transform("mean")
    correlations["within_transition_centered_pearson"] = float(
        centered["centered_g_dot_delta"].corr(centered["centered_loss_change"])
    )
    (args.root / "summary_statistics.json").write_text(
        json.dumps(correlations, indent=2) + "\n"
    )

    lines = [
        "# 과거 task gradient와 실제 누적 AdamW parameter 이동",
        "",
        "## 측정",
        "",
        "각 전환 `C(k-1) → Ck`에서 active LIBERO action head 전체에 대해",
        "`g_old(theta_pre)^T (theta_post - theta_pre)`를 계산했다.",
        "Gradient는 80개 고정 probe와 고정 noise/time 4회 반복의 평균이다.",
        "Parameter 이동은 10,000 AdamW step으로 만들어진 checkpoint의 정확한 차이다.",
        "",
        "## 주 결과",
        "",
        f"- 방향 충돌 셀: **{correlations['negative_optimization_update_cosine_cells']}/15**.",
        f"- 실제 old-task loss 증가: **{correlations['positive_actual_loss_change_cells']}/15**.",
        f"- 1차 예측과 실제 loss 변화의 부호 일치: **{100*sign_agreement:.1f}%**.",
        f"- 행동 망각 6개 중 `g^T delta > 0`인 셀: **{correlations['behavioral_forgetting_cells_with_positive_g_dot_delta']}/6**.",
        f"- `|cos(-g, delta)|` 최댓값: **{correlations['max_absolute_optimization_update_cosine']:.4f}**.",
        f"- `Delta L - g^T delta`는 **{correlations['positive_higher_order_remainder_cells']}/15**에서 양수였다.",
        f"- Pearson(`g^T delta`, 실제 loss 변화): **{correlations['predicted_vs_actual_loss_pearson']:+.3f}**.",
        f"- Spearman(`g^T delta`, 실제 loss 변화): **{correlations['predicted_vs_actual_loss_spearman']:+.3f}**.",
        "",
        "높은 상관은 셀 사이의 상대적 크기 순서를 일부 추적한다는 뜻이다. 그러나 절대 부호를",
        "대부분 틀렸고 모든 행동 망각 셀에서 개선 방향을 예측했으므로 직접적인 망각 설명은 아니다.",
        "",
        "## 셀별 값",
        "",
        "| 전환 | 과거 task | cos(-g, delta) | g^T delta | 실제 loss 변화 | 성공률 변화(pp) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| C{int(row.pre_checkpoint)}→C{int(row.post_checkpoint)} "
            f"| T{int(row.old_task)} | {row.optimization_update_cosine:+.4f} "
            f"| {row.g_dot_delta:+.5f} | {row.actual_loss_change_mean:+.5f} "
            f"| {100*row.success_change:+.1f} |"
        )
    lines += [
        "",
        "## 해석",
        "",
        "Endpoint에서 기존 task 최적화 방향과 실제 누적 이동은 사실상 직교했다.",
        "시작점 gradient의 1차 항만으로 10,000-step 뒤의 loss 증가와 행동 망각을 설명할 수 없다.",
        "`Delta L - g^T delta`가 전 셀에서 양수라는 결과는 학습 중 gradient 변화와 parameter-space",
        "곡률을 포함한 높은 차수 효과가 크다는 것을 보여준다.",
        "",
        "이 결과는 학습 도중 순간적인 conflict가 없었다는 뜻은 아니다. 중간 training state가 저장되지 않아",
        "과거의 step별 AdamW 방향은 복원할 수 없다. 이를 검증하려면 재학습하면서 optimizer step 직전 old",
        "gradient와 실제 한 step 이동을 기록해야 한다.",
        "",
        "15개 셀은 하나의 training seed를 공유하므로 독립적인 유의성 표본이 아니다.",
        "모듈별 분해는 주 결과에서 제외했다.",
    ]
    (args.root / "RESULTS_KO.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
