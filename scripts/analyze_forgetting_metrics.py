#!/usr/bin/env python3
"""Measure four complementary continual-forgetting signals.

For every transition k-1 -> k and old task j < k this script reports:

1. actual action-head parameter displacement between the two checkpoints,
2. fixed-probe old-task flow-loss change,
3. deterministic action-head velocity-output drift, and
4. rollout-success change.

The action-output comparison reuses exactly the same fixed demonstrations and
resets all RNGs before every batch at every checkpoint.  Consequently, the two
models receive the same Gaussian flow noise and the same sampled flow time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

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
        "--gradient-phase-csv",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/gradient_phase_change_seed42/"
            "gradient_phase_change.csv"
        ),
        help="Existing fixed-probe loss results; used to cross-check new forward losses.",
    )
    parser.add_argument(
        "--success-matrix",
        type=Path,
        default=Path(
            "outputs/eval_output/libero_cl/libero_spatial/none/"
            "continual_metrics_task0_to_9/success_matrix.csv"
        ),
    )
    parser.add_argument("--transitions", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/forgetting_metrics_seed42"
        ),
    )
    parser.add_argument(
        "--parameter-only",
        action="store_true",
        help="Only compute checkpoint parameter displacement; no model forward passes.",
    )
    return parser.parse_args()


def checkpoint_path(args: argparse.Namespace, task: int) -> Path:
    return args.experiment_root / f"task_{task}" / f"checkpoint-{args.checkpoint_step}"


def read_weight_map(checkpoint: Path) -> dict[str, str]:
    index_path = checkpoint / "model.safetensors.index.json"
    with index_path.open() as f:
        return json.load(f)["weight_map"]


def parameter_displacement(pre: Path, post: Path) -> list[dict[str, object]]:
    """Stream checkpoint tensors and aggregate exact module-wise L2 statistics."""
    pre_map = read_weight_map(pre)
    post_map = read_weight_map(post)
    action_names = sorted(name for name in pre_map if name.startswith("action_head."))
    if set(action_names) != {name for name in post_map if name.startswith("action_head.")}:
        raise ValueError(f"Action-head tensor sets differ: {pre} vs {post}")

    stats = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    maxima = defaultdict(float)
    with ExitStack() as stack:
        pre_handles = {
            shard: stack.enter_context(safe_open(pre / shard, framework="pt", device="cpu"))
            for shard in sorted(set(pre_map[name] for name in action_names))
        }
        post_handles = {
            shard: stack.enter_context(safe_open(post / shard, framework="pt", device="cpu"))
            for shard in sorted(set(post_map[name] for name in action_names))
        }
        for name in action_names:
            group = parameter_group(name)
            if group is None:
                continue
            before = pre_handles[pre_map[name]].get_tensor(name).float()
            after = post_handles[post_map[name]].get_tensor(name).float()
            if before.shape != after.shape:
                raise ValueError(f"Shape mismatch for {name}: {before.shape} vs {after.shape}")
            delta = after - before
            numel = delta.numel()
            delta_sq = float(torch.sum(delta.double().square()).item())
            before_sq = float(torch.sum(before.double().square()).item())
            after_sq = float(torch.sum(after.double().square()).item())
            delta_l1 = float(torch.sum(delta.double().abs()).item())
            delta_max = float(torch.max(delta.abs()).item()) if numel else 0.0
            for aggregate in (group, "all_action_head"):
                stats[aggregate]["delta_sq"] += delta_sq
                stats[aggregate]["before_sq"] += before_sq
                stats[aggregate]["after_sq"] += after_sq
                stats[aggregate]["delta_l1"] += delta_l1
                counts[aggregate] += numel
                maxima[aggregate] = max(maxima[aggregate], delta_max)
            del before, after, delta

    rows = []
    for group in GROUP_ORDER:
        delta_norm = math.sqrt(stats[group]["delta_sq"])
        before_norm = math.sqrt(stats[group]["before_sq"])
        after_norm = math.sqrt(stats[group]["after_sq"])
        rows.append(
            {
                "module": group,
                "num_parameters": counts[group],
                "parameter_delta_norm": delta_norm,
                "parameter_pre_norm": before_norm,
                "parameter_post_norm": after_norm,
                "relative_parameter_displacement": (
                    delta_norm / before_norm if before_norm > 0 else float("nan")
                ),
                "parameter_delta_mean_abs": stats[group]["delta_l1"] / counts[group],
                "parameter_delta_max_abs": maxima[group],
            }
        )
    return rows


def collect_fixed_outputs(
    model: torch.nn.Module,
    samples: list[dict],
    collator: DefaultDataCollator,
    batch_size: int,
    seed: int,
    device_type: str,
) -> dict[str, object]:
    """Collect deterministic training-time predicted velocities and loss."""
    predicted_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = len(samples)
    hook_output: dict[str, torch.Tensor] = {}

    def capture_decoder_output(_module, _inputs, output):
        hook_output["prediction"] = output.detach()

    handle = model.action_head.action_decoder.register_forward_hook(capture_decoder_output)
    try:
        with torch.no_grad():
            for batch_index, sample_batch in enumerate(iter_batches(samples, batch_size)):
                set_batch_rng(seed, batch_index)
                batch = collator(sample_batch)
                hook_output.clear()
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    result = model(batch)
                prediction = hook_output["prediction"][:, -batch["action"].shape[1] :]
                mask = batch["action_mask"]
                predicted_batches.append(prediction.float().cpu())
                mask_batches.append(mask.float().cpu())
                total_loss += (
                    float(result["loss"].detach().float().item())
                    * len(sample_batch)
                    / total_samples
                )
                del batch, result, prediction, mask
    finally:
        handle.remove()

    return {
        "loss": total_loss,
        "prediction": torch.cat(predicted_batches),
        "mask": torch.cat(mask_batches),
    }


def output_drift(before: dict[str, object], after: dict[str, object]) -> dict[str, float]:
    pre = before["prediction"]
    post = after["prediction"]
    pre_mask = before["mask"]
    post_mask = after["mask"]
    if pre.shape != post.shape or not torch.equal(pre_mask, post_mask):
        raise ValueError("Fixed-probe predictions or masks do not align")
    mask = pre_mask
    delta = (post - pre) * mask
    pre_masked = pre * mask
    post_masked = post * mask
    count = float(mask.sum().item())
    delta_sq = float(torch.sum(delta.double().square()).item())
    pre_sq = float(torch.sum(pre_masked.double().square()).item())
    post_sq = float(torch.sum(post_masked.double().square()).item())
    dot = float(torch.sum(pre_masked.double() * post_masked.double()).item())
    delta_norm = math.sqrt(delta_sq)
    pre_norm = math.sqrt(pre_sq)
    post_norm = math.sqrt(post_sq)
    return {
        "velocity_output_rmse": math.sqrt(delta_sq / count),
        "velocity_output_delta_norm": delta_norm,
        "relative_velocity_output_drift": (
            delta_norm / pre_norm if pre_norm > 0 else float("nan")
        ),
        "velocity_output_cosine": (
            dot / (pre_norm * post_norm) if pre_norm > 0 and post_norm > 0 else float("nan")
        ),
        "velocity_output_elements": int(count),
    }


def read_success(path: Path) -> dict[tuple[int, int], float]:
    values = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            checkpoint = int(row["checkpoint"].removeprefix("checkpoint_task_"))
            for task in range(10):
                value = row.get(f"eval_task_{task}", "")
                if value:
                    values[(checkpoint, task)] = float(value)
    return values


def read_existing_losses(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    values = {}
    if not path.exists():
        return values
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["module"] != "all_action_head":
                continue
            values[(int(row["transition"]), int(row["old_task"]))] = {
                "existing_pre_old_loss": float(row["pre_old_loss"]),
                "existing_post_old_loss": float(row["post_old_loss"]),
                "existing_old_loss_change": float(row["old_loss_change"]),
            }
    return values


def main() -> None:
    args = parse_args()
    transitions = sorted(set(args.transitions))
    if any(not 1 <= transition <= 9 for transition in transitions):
        raise ValueError(f"Transitions must be in [1, 9], got {transitions}")
    args.output.mkdir(parents=True, exist_ok=True)

    parameter_rows: list[dict[str, object]] = []
    parameter_by_transition: dict[int, dict[str, dict[str, object]]] = {}
    for transition in transitions:
        print(f"[transition {transition}] streaming parameter displacement", flush=True)
        rows = parameter_displacement(
            checkpoint_path(args, transition - 1), checkpoint_path(args, transition)
        )
        parameter_by_transition[transition] = {}
        for row in rows:
            enriched = {"transition": transition, **row}
            parameter_rows.append(enriched)
            parameter_by_transition[transition][str(row["module"])] = enriched
        write_rows(args.output / "parameter_displacement_by_module.csv", parameter_rows)

    if args.parameter_only:
        print(f"Wrote {len(parameter_rows)} parameter rows to {args.output}")
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    manifest = read_manifest(args.probe_manifest)
    relevant_checkpoints = sorted(set(transitions) | {transition - 1 for transition in transitions})
    needed_tasks = {
        checkpoint: sorted(
            {
                old_task
                for transition in transitions
                for old_task in range(transition)
                if checkpoint in {transition - 1, transition}
            }
        )
        for checkpoint in relevant_checkpoints
    }
    output_bank: dict[tuple[int, int], dict[str, object]] = {}
    sample_cache: dict[int, list[dict]] = {}
    collator = DefaultDataCollator()
    device_type = torch.device(args.device).type

    def samples_for(task: int) -> list[dict]:
        if task not in sample_cache:
            rows = manifest[task]
            dataset = make_dataset(rows[0])
            sample_cache[task] = materialize_samples(dataset, rows)
            del dataset
        return sample_cache[task]

    for checkpoint_task in relevant_checkpoints:
        print(f"[checkpoint task {checkpoint_task}] loading model", flush=True)
        model = load_model(checkpoint_path(args, checkpoint_task), args.device)
        for old_task in needed_tasks[checkpoint_task]:
            samples = samples_for(old_task)
            print(
                f"  fixed task {old_task}: {len(samples)} samples, deterministic velocity output",
                flush=True,
            )
            output_bank[(checkpoint_task, old_task)] = collect_fixed_outputs(
                model,
                samples,
                collator,
                args.batch_size,
                args.analysis_seed,
                device_type,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    success = read_success(args.success_matrix)
    existing_losses = read_existing_losses(args.gradient_phase_csv)
    combined_rows: list[dict[str, object]] = []
    for transition in transitions:
        parameter = parameter_by_transition[transition]["all_action_head"]
        for old_task in range(transition):
            before = output_bank[(transition - 1, old_task)]
            after = output_bank[(transition, old_task)]
            pre_loss = float(before["loss"])
            post_loss = float(after["loss"])
            success_before = success.get((transition - 1, old_task), float("nan"))
            success_after = success.get((transition, old_task), float("nan"))
            row = {
                "analysis_seed": args.analysis_seed,
                "transition": transition,
                "pre_checkpoint_task": transition - 1,
                "post_checkpoint_task": transition,
                "old_task": old_task,
                "num_probe_samples": len(manifest[old_task]),
                "parameter_delta_norm": parameter["parameter_delta_norm"],
                "relative_parameter_displacement": parameter[
                    "relative_parameter_displacement"
                ],
                "parameter_delta_mean_abs": parameter["parameter_delta_mean_abs"],
                "pre_old_loss": pre_loss,
                "post_old_loss": post_loss,
                "old_loss_change": post_loss - pre_loss,
                **output_drift(before, after),
                "success_before": success_before,
                "success_after": success_after,
                "success_change": success_after - success_before,
                "forgetting_amount": success_before - success_after,
                **existing_losses.get((transition, old_task), {}),
            }
            combined_rows.append(row)
    write_rows(args.output / "combined_forgetting_metrics.csv", combined_rows)

    metadata = {
        "analysis_seed": args.analysis_seed,
        "transitions": transitions,
        "checkpoint_step": args.checkpoint_step,
        "batch_size": args.batch_size,
        "num_pairs": len(combined_rows),
        "parameter_displacement": (
            "exact checkpoint difference over every action_head tensor; module-wise L2"
        ),
        "old_task_loss": (
            "mean flow-matching loss on fixed probes with identical RNG per checkpoint"
        ),
        "action_output_drift": (
            "masked drift of action_decoder predicted flow velocity under identical demo, "
            "Gaussian noise, and flow time; this is not free-running sampled action drift"
        ),
        "rollout_success": str(args.success_matrix.resolve()),
        "precision": "BF16 autocast for model forward; metrics accumulated in FP64",
    }
    with (args.output / "analysis_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
