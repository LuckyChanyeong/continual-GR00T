#!/usr/bin/env python3
"""Compare old-task gradients with actual full-phase AdamW displacement.

For each transition k-1 -> k and old task j < k, compute

    g_j(theta_{k-1})^T (theta_k - theta_{k-1})

on the fixed gradient_first_v1 probes/noise. The checkpoint difference is the
actual cumulative parameter displacement produced by 10,000 AdamW steps. It is
not a reconstruction of any individual optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from analyze_demo_gradients import write_rows
from run_gradient_first import (
    CategorySpecificLinear,
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
    capture,
    checkpoint,
    load_head_weights,
    loss_gradient,
    parameter_group,
    weight_map,
)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(
            "outputs/libero_cl/libero_spatial/none/"
            "step10000_bs16_adamw_bnb_8bits42"
        ),
    )
    parser.add_argument(
        "--source-analysis",
        type=Path,
        default=Path("outputs/analysis/libero_spatial/none/gradient_first_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/libero_spatial/none/update_alignment_v1"),
    )
    parser.add_argument("--transitions", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def active_category_parameters(head: FlowmatchingActionHead) -> set[str]:
    result = set()
    for module_name, module in head.named_modules():
        if isinstance(module, CategorySpecificLinear):
            result.add("action_head." + module_name + ".W")
            result.add("action_head." + module_name + ".b")
    return result


def load_active_delta(
    pre: Path,
    post: Path,
    category_names: set[str],
    embodiment: int,
) -> tuple[dict[str, torch.Tensor], float, dict[str, object]]:
    """Load theta_post-theta_pre, slicing category-specific tensors to LIBERO."""
    pre_map = weight_map(pre)
    post_map = weight_map(post)
    names = sorted(name for name in pre_map if parameter_group(name) is not None)
    post_names = {name for name in post_map if parameter_group(name) is not None}
    if set(names) != post_names:
        raise RuntimeError("Pre/post action-head parameter names differ")

    result: dict[str, torch.Tensor] = {}
    squared_norm = 0.0
    inactive_exact = True
    sliced_parameter_count = 0
    active_numel = 0
    with ExitStack() as stack:
        pre_handles = {
            shard: stack.enter_context(safe_open(pre / shard, framework="pt", device="cpu"))
            for shard in sorted(set(pre_map[name] for name in names))
        }
        post_handles = {
            shard: stack.enter_context(safe_open(post / shard, framework="pt", device="cpu"))
            for shard in sorted(set(post_map[name] for name in names))
        }
        for name in names:
            before = pre_handles[pre_map[name]].get_tensor(name)
            after = post_handles[post_map[name]].get_tensor(name)
            if name in category_names:
                if embodiment:
                    inactive_exact &= torch.equal(before[:embodiment], after[:embodiment])
                if embodiment + 1 < before.shape[0]:
                    inactive_exact &= torch.equal(before[embodiment + 1 :], after[embodiment + 1 :])
                before = before[embodiment]
                after = after[embodiment]
                sliced_parameter_count += 1
            value = (after.float() - before.float()).contiguous()
            result[name] = value
            active_numel += value.numel()
            squared_norm += float(torch.sum(value * value, dtype=torch.float64))
    audit = {
        "inactive_category_slices_exactly_unchanged": bool(inactive_exact),
        "category_specific_parameter_tensors_sliced": sliced_parameter_count,
        "active_parameter_numel": active_numel,
    }
    if not inactive_exact:
        raise RuntimeError("An inactive embodiment slice changed across checkpoints")
    return result, math.sqrt(squared_norm), audit


def project(
    gradient: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    delta_norm: float,
) -> dict[str, float]:
    if set(gradient) != set(delta):
        missing_gradient = sorted(set(delta) - set(gradient))
        missing_delta = sorted(set(gradient) - set(delta))
        raise RuntimeError(
            f"Gradient/delta names differ: gradient_missing={missing_gradient[:3]}, "
            f"delta_missing={missing_delta[:3]}"
        )
    dot = 0.0
    gradient_squared = 0.0
    for name in sorted(gradient):
        grad = gradient[name]
        displacement = delta[name]
        if grad.shape != displacement.shape:
            raise RuntimeError(f"Shape mismatch for {name}: {grad.shape} vs {displacement.shape}")
        dot += float(torch.sum(grad * displacement, dtype=torch.float64))
        gradient_squared += float(torch.sum(grad * grad, dtype=torch.float64))
    gradient_norm = math.sqrt(gradient_squared)
    denominator = gradient_norm * delta_norm
    gradient_delta_cosine = dot / denominator if denominator else float("nan")
    return {
        "g_dot_delta": dot,
        "gradient_norm": gradient_norm,
        "parameter_delta_norm": delta_norm,
        "gradient_delta_cosine": gradient_delta_cosine,
        "optimization_update_cosine": -gradient_delta_cosine,
    }


def pooled_gradient(
    head: FlowmatchingActionHead,
    data: dict,
    repeat: int,
    batch_size: int,
) -> tuple[float, dict[str, torch.Tensor]]:
    bank_gradients = []
    bank_losses = []
    embodiment = int(data["embodiment_id"][0])
    for bank in ("A", "B"):
        indices = [index for index, label in enumerate(data["banks"]) if label == bank]
        bank_losses.append(loss_gradient(head, data, indices, repeat, batch_size))
        bank_gradients.append(capture(head, embodiment))
    pooled = bank_gradients[0]
    right = bank_gradients[1]
    for name in pooled:
        pooled[name].mul_(0.5).add_(right[name], alpha=0.5)
    del right, bank_gradients
    return float(np.mean(bank_losses)), pooled


def update_mean(
    current: dict[str, torch.Tensor] | None,
    value: dict[str, torch.Tensor],
    repeat: int,
) -> dict[str, torch.Tensor]:
    if current is None:
        return value
    old_weight = repeat / (repeat + 1)
    new_weight = 1 / (repeat + 1)
    for name in current:
        current[name].mul_(old_weight).add_(value[name], alpha=new_weight)
    return current


def load_phase_losses(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    grouped: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["bank"] != "all":
                continue
            key = (int(row["transition"]), int(row["old_task"]))
            grouped.setdefault(key, []).append(
                (float(row["pre_loss"]), float(row["post_loss"]), float(row["loss_change"]))
            )
    result = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        result[key] = {
            "pre_loss_mean": float(array[:, 0].mean()),
            "post_loss_mean": float(array[:, 1].mean()),
            "actual_loss_change_mean": float(array[:, 2].mean()),
            "actual_loss_change_std": float(array[:, 2].std(ddof=1)),
        }
    return result


def load_success(path: Path) -> dict[tuple[int, int], float]:
    result = {}
    with path.open(newline="") as handle:
        for checkpoint_index, row in enumerate(csv.DictReader(handle)):
            for task in range(6):
                value = row.get(f"eval_task_{task}", "")
                if value:
                    result[(checkpoint_index, task)] = float(value)
    return result


def main() -> None:
    args = args_parser()
    args.output.mkdir(parents=True, exist_ok=True)
    transitions = sorted(set(args.transitions))
    if any(transition not in range(1, 6) for transition in transitions):
        raise ValueError(f"Transitions must be within 1..5, got {transitions}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    config = json.loads((checkpoint(args.experiment_root, 0) / "config.json").read_text())
    head = FlowmatchingActionHead(FlowmatchingActionHeadConfig(**config["action_head_cfg"]))
    head = head.float().cuda().eval()
    category_names = active_category_parameters(head)

    sample_cache = torch.load(args.source_analysis / "cache/task_0.pt", weights_only=False)
    embodiment = int(sample_cache["embodiment_id"][0])
    del sample_cache

    phase_losses = load_phase_losses(args.source_analysis / "phase_losses.csv")
    success = load_success(args.source_analysis / "figures/05_continual_success_task0_to_5.csv")
    repeat_rows = []
    mean_rows = []
    delta_audits = []

    for transition in transitions:
        pre = checkpoint(args.experiment_root, transition - 1)
        post = checkpoint(args.experiment_root, transition)
        print(f"TRANSITION {transition}: loading active checkpoint displacement", flush=True)
        delta, delta_norm, delta_audit = load_active_delta(
            pre, post, category_names, embodiment
        )
        delta_audits.append({"transition": transition, **delta_audit, "delta_norm": delta_norm})
        load_head_weights(head, pre)

        for old_task in range(transition):
            print(f"  OLD TASK {old_task}: {args.repeats} fixed-noise gradients", flush=True)
            data = torch.load(
                args.source_analysis / f"cache/task_{old_task}.pt", weights_only=False
            )
            if int(data["embodiment_id"][0]) != embodiment:
                raise RuntimeError("Embodiment changed between cached tasks")
            mean_gradient = None
            computed_losses = []
            repeat_dots = []
            for repeat in range(args.repeats):
                loss, gradient = pooled_gradient(head, data, repeat, args.batch_size)
                metrics = project(gradient, delta, delta_norm)
                computed_losses.append(loss)
                repeat_dots.append(metrics["g_dot_delta"])
                repeat_rows.append(
                    {
                        "transition": transition,
                        "old_task": old_task,
                        "repeat": repeat,
                        "pre_loss": loss,
                        **metrics,
                    }
                )
                mean_gradient = update_mean(mean_gradient, gradient, repeat)
                if mean_gradient is not gradient:
                    del gradient
                write_rows(args.output / "update_alignment_repeats.csv", repeat_rows)
                gc.collect()

            assert mean_gradient is not None
            metrics = project(mean_gradient, delta, delta_norm)
            existing = phase_losses[(transition, old_task)]
            if not math.isclose(
                float(np.mean(computed_losses)), existing["pre_loss_mean"], rel_tol=1e-5, abs_tol=1e-6
            ):
                raise RuntimeError(
                    f"Pre-loss mismatch at transition {transition}, old task {old_task}: "
                    f"{np.mean(computed_losses)} vs {existing['pre_loss_mean']}"
                )
            if not math.isclose(
                metrics["g_dot_delta"], float(np.mean(repeat_dots)), rel_tol=1e-6, abs_tol=1e-8
            ):
                raise RuntimeError("Mean-gradient projection failed linearity check")
            before_success = success[(transition - 1, old_task)]
            after_success = success[(transition, old_task)]
            mean_rows.append(
                {
                    "transition": transition,
                    "old_task": old_task,
                    "pre_checkpoint": transition - 1,
                    "post_checkpoint": transition,
                    "num_episodes": 20,
                    "num_probes": 80,
                    "noise_repeats": args.repeats,
                    **metrics,
                    "g_dot_delta_repeat_std": float(np.std(repeat_dots, ddof=1)),
                    **existing,
                    "success_before": before_success,
                    "success_after": after_success,
                    "success_change": after_success - before_success,
                }
            )
            write_rows(args.output / "update_alignment.csv", mean_rows)
            print(
                f"    gTd={metrics['g_dot_delta']:+.6g} "
                f"cos(-g,d)={metrics['optimization_update_cosine']:+.4f} "
                f"dLoss={existing['actual_loss_change_mean']:+.4f} "
                f"dSuccess={after_success-before_success:+.3f}",
                flush=True,
            )
            del data, mean_gradient
            head.zero_grad(set_to_none=True)
            gc.collect()

        del delta
        gc.collect()

    metadata = {
        "status": "complete",
        "definition": "g_old(theta_pre)^T @ (theta_post - theta_pre)",
        "scope": "active LIBERO action-head parameters",
        "parameter_displacement": "actual cumulative displacement across 10,000 AdamW optimizer steps",
        "not_measured": "individual historical AdamW step directions; training-trajectory checkpoints were not saved",
        "tasks": list(range(6)),
        "transitions": transitions,
        "episodes_per_task": 20,
        "probes_per_task": 80,
        "noise_time_repeats": args.repeats,
        "source_analysis": str(args.source_analysis.resolve()),
        "experiment_root": str(args.experiment_root.resolve()),
        "delta_audits": delta_audits,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(args.output / "analysis_metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
