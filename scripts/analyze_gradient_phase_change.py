#!/usr/bin/env python3
"""Compare old-task flow gradients before and after every continual phase.

For transition k and every old task j < k, this computes three scale-invariant
cosines from fixed demonstration samples:

1. old_drift:      cos(g[k-1,j], g[k,j])
2. cross_endpoint: cos(g[k-1,j], g[k,k])
3. post_conflict:  cos(g[k,j],   g[k,k])

Raw gradient magnitudes are retained separately as norms and norm ratios.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import torch

from analyze_demo_gradients import (
    GROUP_ORDER,
    capture_gradients_cpu,
    compare_with_reference,
    compute_mean_gradient,
    group_squared_norms,
    load_model,
    make_dataset,
    materialize_samples,
    read_manifest,
    write_rows,
)
from gr00t.model.transforms import DefaultDataCollator


BASE_MODULES = [group for group in GROUP_ORDER if group != "all_action_head"]


def parse_args() -> argparse.Namespace:
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
        "--probe-manifest",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/probe_bank/probe_manifest.csv"
        ),
    )
    parser.add_argument(
        "--success-matrix",
        type=Path,
        default=Path(
            "outputs/eval_output/libero_cl/libero_spatial/none/"
            "continual_metrics_task0_to_9/success_matrix.csv"
        ),
    )
    parser.add_argument("--transitions", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/gradient_phase_change_seed42"
        ),
    )
    return parser.parse_args()


def read_success_matrix(path: Path) -> dict[tuple[int, int], float]:
    result = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            checkpoint = int(row["checkpoint"].removeprefix("checkpoint_task_"))
            for task in range(10):
                value = row.get(f"eval_task_{task}", "")
                if value:
                    result[(checkpoint, task)] = float(value)
    return result


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def safe_log10(value: float) -> float:
    return math.log10(value) if value > 0 else float("nan")


def angle_degrees(cosine: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def main() -> None:
    args = parse_args()
    transitions = sorted(set(args.transitions))
    if any(not 1 <= transition <= 9 for transition in transitions):
        raise ValueError(f"Transitions must be in [1, 9], got {transitions}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # Reduce run-to-run numerical variation. Some third-party attention kernels
    # may only provide a warning rather than a deterministic implementation.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    manifest = read_manifest(args.probe_manifest)
    success = read_success_matrix(args.success_matrix)
    collator = DefaultDataCollator()
    device_type = torch.device(args.device).type
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for transition in transitions:
        task_cache: dict[int, list[dict]] = {}

        def samples_for(task_id: int) -> list[dict]:
            if task_id not in task_cache:
                task_rows = manifest[task_id]
                dataset = make_dataset(task_rows[0])
                task_cache[task_id] = materialize_samples(dataset, task_rows)
                del dataset
            return task_cache[task_id]

        pre_checkpoint = (
            args.experiment_root
            / f"task_{transition - 1}"
            / f"checkpoint-{args.checkpoint_step}"
        )
        post_checkpoint = (
            args.experiment_root
            / f"task_{transition}"
            / f"checkpoint-{args.checkpoint_step}"
        )

        print(f"[transition {transition}] PRE checkpoint task {transition - 1}", flush=True)
        pre_model = load_model(pre_checkpoint, args.device)
        pre_old: dict[int, dict[str, object]] = {}
        for old_task in range(transition):
            samples = samples_for(old_task)
            print(f"  pre old task {old_task}: {len(samples)} samples", flush=True)
            loss = compute_mean_gradient(
                pre_model,
                samples,
                collator,
                args.batch_size,
                args.analysis_seed,
                device_type,
            )
            pre_old[old_task] = {
                "loss": loss,
                "norms": group_squared_norms(pre_model),
                "gradient": capture_gradients_cpu(pre_model),
            }

        del pre_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[transition {transition}] POST checkpoint task {transition}", flush=True)
        post_model = load_model(post_checkpoint, args.device)
        new_samples = samples_for(transition)
        post_new_loss = compute_mean_gradient(
            post_model,
            new_samples,
            collator,
            args.batch_size,
            args.analysis_seed,
            device_type,
        )
        post_new_norms = group_squared_norms(post_model)

        # Compare every pre-old gradient to the same post-new gradient while it
        # is resident on GPU, then retain post-new once for later post comparisons.
        cross_comparisons = {
            old_task: compare_with_reference(
                post_model,
                pre_old[old_task]["gradient"],
                pre_old[old_task]["norms"],
            )
            for old_task in range(transition)
        }
        post_new_gradient = capture_gradients_cpu(post_model)

        for old_task in range(transition):
            old_samples = samples_for(old_task)
            print(f"  post old task {old_task}: {len(old_samples)} samples", flush=True)
            post_old_loss = compute_mean_gradient(
                post_model,
                old_samples,
                collator,
                args.batch_size,
                args.analysis_seed,
                device_type,
            )
            post_old_norms = group_squared_norms(post_model)

            drift = compare_with_reference(
                post_model,
                pre_old[old_task]["gradient"],
                pre_old[old_task]["norms"],
            )
            post_conflict = compare_with_reference(
                post_model, post_new_gradient, post_new_norms
            )
            cross = cross_comparisons[old_task]
            pre_old_loss = float(pre_old[old_task]["loss"])
            success_before = success.get((transition - 1, old_task), float("nan"))
            success_after = success.get((transition, old_task), float("nan"))

            module_metrics = {}
            for module in GROUP_ORDER:
                pre_norm = float(pre_old[old_task]["norms"][module])
                post_old_norm = post_old_norms[module]
                post_new_norm = post_new_norms[module]
                norm_ratio = safe_ratio(post_old_norm, pre_norm)
                row = {
                    "analysis_seed": args.analysis_seed,
                    "transition": transition,
                    "pre_checkpoint_task": transition - 1,
                    "post_checkpoint_task": transition,
                    "old_task": old_task,
                    "new_task": transition,
                    "module": module,
                    "num_samples_per_task": len(old_samples),
                    "old_drift_cosine": drift[module]["cosine"],
                    "old_drift_angle_degrees": angle_degrees(drift[module]["cosine"]),
                    "old_direction_distance": 1.0 - drift[module]["cosine"],
                    "cross_endpoint_cosine": cross[module]["cosine"],
                    "post_old_new_cosine": post_conflict[module]["cosine"],
                    "pre_old_norm": pre_norm,
                    "post_old_norm": post_old_norm,
                    "post_new_norm": post_new_norm,
                    "old_norm_ratio": norm_ratio,
                    "old_log10_norm_ratio": safe_log10(norm_ratio),
                    "pre_old_loss": pre_old_loss,
                    "post_old_loss": post_old_loss,
                    "post_new_loss": post_new_loss,
                    "old_loss_change": post_old_loss - pre_old_loss,
                    "success_before": success_before,
                    "success_after": success_after,
                    "success_change": success_after - success_before,
                }
                output_rows.append(row)
                module_metrics[module] = row

            full = module_metrics["all_action_head"]
            pair_rows.append(
                {
                    "analysis_seed": args.analysis_seed,
                    "transition": transition,
                    "old_task": old_task,
                    "new_task": transition,
                    "global_old_drift_cosine": full["old_drift_cosine"],
                    "macro_old_drift_cosine": sum(
                        module_metrics[m]["old_drift_cosine"] for m in BASE_MODULES
                    ) / len(BASE_MODULES),
                    "global_cross_endpoint_cosine": full["cross_endpoint_cosine"],
                    "macro_cross_endpoint_cosine": sum(
                        module_metrics[m]["cross_endpoint_cosine"] for m in BASE_MODULES
                    ) / len(BASE_MODULES),
                    "global_post_old_new_cosine": full["post_old_new_cosine"],
                    "macro_post_old_new_cosine": sum(
                        module_metrics[m]["post_old_new_cosine"] for m in BASE_MODULES
                    ) / len(BASE_MODULES),
                    "global_old_norm_ratio": full["old_norm_ratio"],
                    "global_old_log10_norm_ratio": full["old_log10_norm_ratio"],
                    "old_loss_change": full["old_loss_change"],
                    "success_change": full["success_change"],
                }
            )
            write_rows(args.output / "gradient_phase_change.csv", output_rows)
            write_rows(args.output / "pair_summary.csv", pair_rows)

        del post_new_gradient, post_model, pre_old, task_cache
        gc.collect()
        torch.cuda.empty_cache()

    metadata = {
        "analysis_seed": args.analysis_seed,
        "transitions": transitions,
        "checkpoint_step": args.checkpoint_step,
        "batch_size": args.batch_size,
        "probe_manifest": str(args.probe_manifest.resolve()),
        "success_matrix": str(args.success_matrix.resolve()),
        "gradient_definition": "mean fixed-demo flow-matching gradient",
        "scaling": (
            "cosines use per-module L2 normalization; raw magnitude is reported "
            "separately as norms, ratios, and log10 ratios"
        ),
        "comparisons": {
            "old_drift": "cos(g[k-1,j], g[k,j])",
            "cross_endpoint": "cos(g[k-1,j], g[k,k])",
            "post_old_new": "cos(g[k,j], g[k,k])",
        },
        "precision": "BF16 autocast; captured gradients stored on CPU as FP32",
        "determinism": (
            "fixed Python/NumPy/Torch RNG per batch; cuDNN deterministic; TF32 disabled; "
            "torch deterministic algorithms strictly enabled"
        ),
    }
    with (args.output / "analysis_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
