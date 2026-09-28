#!/usr/bin/env python3
"""Task-wise diagonal empirical Fisher and importance-weighted forgetting drift.

The full Hessian of a billion-parameter action head is intractable.  This script
computes the standard diagonal empirical-Fisher approximation at each task's
acquisition checkpoint, saves it, and measures how much later checkpoint motion
occurs along task-important parameter coordinates.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from analyze_demo_gradients import (
    GROUP_ORDER,
    iter_batches,
    load_model,
    make_dataset,
    materialize_samples,
    parameter_group,
    read_manifest,
    set_batch_rng,
    write_rows,
)
from gr00t.model.transforms import DefaultDataCollator


def parse_args():
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
        "--forgetting-csv",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/forgetting_metrics_seed42/"
            "combined_forgetting_metrics.csv"
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
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Use 1 for true sample-wise empirical Fisher; larger values are minibatch approximations.",
    )
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/libero_spatial/none/fisher_forgetting_seed42"),
    )
    parser.add_argument("--reuse-fisher", action="store_true")
    return parser.parse_args()


def checkpoint_path(args, task):
    return args.experiment_root / f"task_{task}" / f"checkpoint-{args.checkpoint_step}"


def weight_map(checkpoint: Path):
    with (checkpoint / "model.safetensors.index.json").open() as f:
        return json.load(f)["weight_map"]


def action_parameters(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter_group(name) is not None and parameter.requires_grad
    ]


def compute_diagonal_fisher(model, samples, collator, batch_size, seed, device_type):
    parameters = action_parameters(model)
    fisher = {
        name: torch.zeros(parameter.shape, dtype=torch.float32, device="cpu")
        for name, parameter in parameters
    }
    batches = list(iter_batches(samples, batch_size))
    total_loss = 0.0
    for batch_index, sample_batch in enumerate(batches):
        set_batch_rng(seed, batch_index)
        model.zero_grad(set_to_none=True)
        batch = collator(sample_batch)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            loss = model(batch)["loss"]
        loss.backward()
        total_loss += float(loss.detach().float().item()) / len(batches)
        with torch.no_grad():
            for name, parameter in parameters:
                if parameter.grad is not None:
                    gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
                    fisher[name].addcmul_(gradient, gradient, value=1.0 / len(batches))
                    del gradient
        print(
            f"    Fisher batch {batch_index + 1}/{len(batches)} loss={float(loss):.6f}",
            flush=True,
        )
        del batch, loss
    model.zero_grad(set_to_none=True)
    return fisher, total_loss


def fisher_statistics(task, fisher):
    sums = defaultdict(float)
    squared = defaultdict(float)
    maxima = defaultdict(float)
    counts = defaultdict(int)
    for name, value in fisher.items():
        group = parameter_group(name)
        value64 = value.double()
        value_sum = float(value64.sum().item())
        value_sq = float((value64 * value64).sum().item())
        value_max = float(value.max().item())
        for aggregate in (group, "all_action_head"):
            sums[aggregate] += value_sum
            squared[aggregate] += value_sq
            maxima[aggregate] = max(maxima[aggregate], value_max)
            counts[aggregate] += value.numel()
    return [
        {
            "task": task,
            "module": group,
            "num_parameters": counts[group],
            "fisher_sum": sums[group],
            "fisher_mean": sums[group] / counts[group],
            "fisher_l2_norm": math.sqrt(squared[group]),
            "fisher_max": maxima[group],
        }
        for group in GROUP_ORDER
    ]


def save_fisher(path: Path, fisher):
    path.parent.mkdir(parents=True, exist_ok=True)
    # BF16 retains the exponent range needed for squared gradients while halving disk use.
    compressed = {name: value.to(torch.bfloat16).contiguous() for name, value in fisher.items()}
    save_file(compressed, path, metadata={"format": "diagonal_empirical_fisher_bfloat16"})
    del compressed


def load_fisher(path: Path):
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).float() for name in handle.keys()}


def importance_weighted_drift(fisher, reference: Path, pre: Path, post: Path):
    maps = {"reference": weight_map(reference), "pre": weight_map(pre), "post": weight_map(post)}
    names = sorted(fisher)
    stats = defaultdict(lambda: defaultdict(float))
    with ExitStack() as stack:
        handles = {}
        for label, checkpoint in (("reference", reference), ("pre", pre), ("post", post)):
            handles[label] = {
                shard: stack.enter_context(
                    safe_open(checkpoint / shard, framework="pt", device="cpu")
                )
                for shard in sorted(set(maps[label][name] for name in names))
            }
        for name in names:
            group = parameter_group(name)
            importance = fisher[name]
            reference_value = handles["reference"][maps["reference"][name]].get_tensor(name).float()
            pre_value = handles["pre"][maps["pre"][name]].get_tensor(name).float()
            post_value = handles["post"][maps["post"][name]].get_tensor(name).float()
            cumulative = post_value - reference_value
            transition = post_value - pre_value
            importance_sum = float(importance.double().sum().item())
            reference_energy = float(
                (importance * reference_value.square()).double().sum().item()
            )
            cumulative_energy = float((importance * cumulative.square()).double().sum().item())
            transition_energy = float((importance * transition.square()).double().sum().item())
            for aggregate in (group, "all_action_head"):
                stats[aggregate]["importance_sum"] += importance_sum
                stats[aggregate]["reference_energy"] += reference_energy
                stats[aggregate]["cumulative_energy"] += cumulative_energy
                stats[aggregate]["transition_energy"] += transition_energy
            del reference_value, pre_value, post_value, cumulative, transition
    rows = []
    for group in GROUP_ORDER:
        values = stats[group]
        fisher_sum = values["importance_sum"]
        reference_energy = values["reference_energy"]
        rows.append(
            {
                "module": group,
                "fisher_weighted_cumulative_rms": math.sqrt(
                    values["cumulative_energy"] / fisher_sum
                ),
                "fisher_weighted_transition_rms": math.sqrt(
                    values["transition_energy"] / fisher_sum
                ),
                "relative_fisher_cumulative_drift": math.sqrt(
                    values["cumulative_energy"] / reference_energy
                ),
                "relative_fisher_transition_drift": math.sqrt(
                    values["transition_energy"] / reference_energy
                ),
            }
        )
    return rows


def fisher_overlap(left_path: Path, right_fisher):
    dots = defaultdict(float)
    left_sq = defaultdict(float)
    right_sq = defaultdict(float)
    with safe_open(left_path, framework="pt", device="cpu") as left:
        for name, right_value in right_fisher.items():
            group = parameter_group(name)
            left_value = left.get_tensor(name).float()
            dot = float((left_value * right_value).double().sum().item())
            l2 = float((left_value * left_value).double().sum().item())
            r2 = float((right_value * right_value).double().sum().item())
            for aggregate in (group, "all_action_head"):
                dots[aggregate] += dot
                left_sq[aggregate] += l2
                right_sq[aggregate] += r2
            del left_value
    return {
        group: dots[group] / math.sqrt(left_sq[group] * right_sq[group])
        for group in GROUP_ORDER
    }


def read_success(path: Path):
    values = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            checkpoint = int(row["checkpoint"].removeprefix("checkpoint_task_"))
            for task in range(10):
                value = row.get(f"eval_task_{task}", "")
                if value:
                    values[(checkpoint, task)] = float(value)
    return values


def read_losses(path: Path):
    values = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            transition = int(row["transition"])
            task = int(row["old_task"])
            values[(transition - 1, task)] = float(row["pre_old_loss"])
            values[(transition, task)] = float(row["post_old_loss"])
    return values


def main():
    args = parse_args()
    tasks = sorted(set(args.tasks))
    if not tasks or tasks != list(range(max(tasks) + 1)):
        raise ValueError("Tasks must be a contiguous range starting at 0")
    max_task = max(tasks)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    fisher_dir = args.output / "fisher_diagonals"
    fisher_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    manifest = read_manifest(args.probe_manifest)
    collator = DefaultDataCollator()
    device_type = torch.device(args.device).type
    fisher_stats_rows = []
    drift_rows = []
    overlap_rows = []

    for task in tasks:
        fisher_path = fisher_dir / f"task_{task}_diagonal_fisher.safetensors"
        if args.reuse_fisher and fisher_path.exists():
            print(f"[task {task}] reusing {fisher_path}", flush=True)
            fisher = load_fisher(fisher_path)
            mean_loss = float("nan")
        else:
            print(f"[task {task}] loading acquisition checkpoint and probes", flush=True)
            rows = manifest[task]
            dataset = make_dataset(rows[0])
            samples = materialize_samples(dataset, rows)
            del dataset
            model = load_model(checkpoint_path(args, task), args.device)
            fisher, mean_loss = compute_diagonal_fisher(
                model,
                samples,
                collator,
                args.batch_size,
                args.analysis_seed,
                device_type,
            )
            del model, samples
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[task {task}] saving BF16 diagonal Fisher", flush=True)
            save_fisher(fisher_path, fisher)

        stats = fisher_statistics(task, fisher)
        for row in stats:
            fisher_stats_rows.append({"acquisition_loss": mean_loss, **row})
        write_rows(args.output / "fisher_module_statistics.csv", fisher_stats_rows)

        for previous_task in range(task):
            previous_path = fisher_dir / f"task_{previous_task}_diagonal_fisher.safetensors"
            overlaps = fisher_overlap(previous_path, fisher)
            for module, cosine in overlaps.items():
                overlap_rows.append(
                    {"task_a": previous_task, "task_b": task, "module": module, "fisher_cosine": cosine}
                )
        if overlap_rows:
            write_rows(args.output / "fisher_task_overlap.csv", overlap_rows)

        reference = checkpoint_path(args, task)
        for checkpoint in range(task + 1, max_task + 1):
            print(
                f"[task {task}] Fisher-weighted drift at checkpoint {checkpoint}", flush=True
            )
            module_rows = importance_weighted_drift(
                fisher,
                reference,
                checkpoint_path(args, checkpoint - 1),
                checkpoint_path(args, checkpoint),
            )
            for row in module_rows:
                drift_rows.append(
                    {
                        "reference_task": task,
                        "checkpoint": checkpoint,
                        "transition": checkpoint,
                        **row,
                    }
                )
            write_rows(args.output / "fisher_weighted_drift_by_module.csv", drift_rows)
        del fisher
        gc.collect()

    success = read_success(args.success_matrix)
    losses = read_losses(args.forgetting_csv)
    global_drift = {
        (int(row["checkpoint"]), int(row["reference_task"])): row
        for row in drift_rows
        if row["module"] == "all_action_head"
    }
    combined = []
    for checkpoint in range(1, max_task + 1):
        for task in range(checkpoint):
            previous_best = max(success[(phase, task)] for phase in range(task, checkpoint))
            current_success = success[(checkpoint, task)]
            acquisition_loss = losses[(task, task)]
            current_loss = losses[(checkpoint, task)]
            drift = global_drift[(checkpoint, task)]
            combined.append(
                {
                    "checkpoint": checkpoint,
                    "old_task": task,
                    "success_at_acquisition": success[(task, task)],
                    "best_previous_success": previous_best,
                    "previous_success": success[(checkpoint - 1, task)],
                    "current_success": current_success,
                    "standard_forgetting_score": previous_best - current_success,
                    "transition_success_drop": success[(checkpoint - 1, task)] - current_success,
                    "backward_transfer": current_success - success[(task, task)],
                    "loss_at_acquisition": acquisition_loss,
                    "previous_old_task_loss": losses[(checkpoint - 1, task)],
                    "current_old_task_loss": current_loss,
                    "cumulative_old_task_loss_increase": current_loss - acquisition_loss,
                    "transition_old_task_loss_increase": current_loss - losses[(checkpoint - 1, task)],
                    "relative_fisher_cumulative_drift": drift[
                        "relative_fisher_cumulative_drift"
                    ],
                    "relative_fisher_transition_drift": drift[
                        "relative_fisher_transition_drift"
                    ],
                }
            )
    write_rows(args.output / "four_forgetting_metrics.csv", combined)
    metadata = {
        "tasks": tasks,
        "analysis_seed": args.analysis_seed,
        "fisher_batch_size": args.batch_size,
        "num_probes_per_task": len(manifest[0]),
        "fisher_definition": (
            "mean of squared gradients per sample" if args.batch_size == 1
            else "mean of squared minibatch-mean gradients"
        ),
        "hessian_interpretation": (
            "diagonal empirical Fisher is a positive-semidefinite curvature/Hessian approximation; "
            "the full off-diagonal Hessian is not constructed"
        ),
        "stored_fisher_dtype": "bfloat16",
        "accumulation_dtype": "float32 CPU",
    }
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
