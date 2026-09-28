#!/usr/bin/env python3
"""Project actual checkpoint displacement onto old-task gradients.

For transition k-1 -> k and every old task j < k, this computes the signed
first-order Taylor term

    g_j(theta_{k-1})^T (theta_k - theta_{k-1})

for each trainable action-head module.  Unlike raw task-task gradient cosine,
this uses the displacement that AdamW actually produced over the full phase.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from analyze_demo_gradients import (
    GROUP_ORDER,
    compute_mean_gradient,
    load_model,
    make_dataset,
    materialize_samples,
    parameter_group,
    read_manifest,
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
        default=Path("outputs/analysis/libero_spatial/none/probe_bank/probe_manifest.csv"),
    )
    parser.add_argument("--transitions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/causal_first5_seed42/"
            "module_update_projection"
        ),
    )
    return parser.parse_args()


def checkpoint_path(args: argparse.Namespace, task: int) -> Path:
    return args.experiment_root / f"task_{task}" / f"checkpoint-{args.checkpoint_step}"


def weight_map(checkpoint: Path) -> dict[str, str]:
    with (checkpoint / "model.safetensors.index.json").open() as handle:
        return json.load(handle)["weight_map"]


def load_action_delta(pre: Path, post: Path):
    pre_map = weight_map(pre)
    post_map = weight_map(post)
    names = sorted(name for name in pre_map if parameter_group(name) is not None)
    if set(names) != {name for name in post_map if parameter_group(name) is not None}:
        raise ValueError("Pre/post action-head parameter sets differ")
    delta = {}
    delta_sq = defaultdict(float)
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
            value = (
                post_handles[post_map[name]].get_tensor(name).float()
                - pre_handles[pre_map[name]].get_tensor(name).float()
            ).contiguous()
            delta[name] = value
            squared = float(torch.sum(value * value, dtype=torch.float64).item())
            group = parameter_group(name)
            delta_sq[group] += squared
            delta_sq["all_action_head"] += squared
    return delta, {group: math.sqrt(delta_sq[group]) for group in GROUP_ORDER}


def project_gradient(model, delta, delta_norms):
    dot = defaultdict(float)
    grad_sq = defaultdict(float)
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if group is None or parameter.grad is None:
            continue
        gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
        displacement = delta[name]
        product = gradient * displacement
        value_dot = float(torch.sum(product, dtype=torch.float64).item())
        value_grad_sq = float(torch.sum(gradient * gradient, dtype=torch.float64).item())
        for aggregate in (group, "all_action_head"):
            dot[aggregate] += value_dot
            grad_sq[aggregate] += value_grad_sq
        del gradient, product

    rows = []
    for group in GROUP_ORDER:
        grad_norm = math.sqrt(grad_sq[group])
        delta_norm = delta_norms[group]
        denominator = grad_norm * delta_norm
        rows.append(
            {
                "module": group,
                "g_dot_delta": dot[group],
                "gradient_norm": grad_norm,
                "parameter_delta_norm": delta_norm,
                "gradient_delta_cosine": dot[group] / denominator if denominator > 0 else float("nan"),
                "g_dot_delta_per_delta_norm": dot[group] / delta_norm if delta_norm > 0 else float("nan"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    transitions = sorted(set(args.transitions))
    if any(transition < 1 for transition in transitions):
        raise ValueError(f"Transitions must be positive, got {transitions}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    manifest = read_manifest(args.probe_manifest)
    collator = DefaultDataCollator()
    device_type = torch.device(args.device).type
    rows = []

    for transition in transitions:
        pre = checkpoint_path(args, transition - 1)
        post = checkpoint_path(args, transition)
        print(f"[transition {transition}] loading actual action-head displacement", flush=True)
        delta, delta_norms = load_action_delta(pre, post)
        print(f"[transition {transition}] loading pre-transition model", flush=True)
        model = load_model(pre, args.device)

        for old_task in range(transition):
            task_rows = manifest[old_task]
            dataset = make_dataset(task_rows[0])
            samples = materialize_samples(dataset, task_rows)
            del dataset
            print(
                f"  old task {old_task}: mean gradient over {len(samples)} fixed probes",
                flush=True,
            )
            loss = compute_mean_gradient(
                model,
                samples,
                collator,
                args.batch_size,
                args.analysis_seed,
                device_type,
            )
            for projection in project_gradient(model, delta, delta_norms):
                rows.append(
                    {
                        "analysis_seed": args.analysis_seed,
                        "transition": transition,
                        "old_task": old_task,
                        "pre_checkpoint": transition - 1,
                        "post_checkpoint": transition,
                        "num_probes": len(samples),
                        "pre_old_task_loss": loss,
                        **projection,
                    }
                )
            write_rows(args.output / "module_g_dot_delta.csv", rows)
            del samples

        model.zero_grad(set_to_none=True)
        del model, delta
        gc.collect()
        torch.cuda.empty_cache()

    metadata = {
        "definition": "g_old(theta_pre)^T * (theta_post - theta_pre)",
        "positive_interpretation": "first-order old-task loss increase",
        "negative_interpretation": "first-order old-task loss decrease",
        "optimizer_awareness": "uses the actual full-phase AdamW checkpoint displacement",
        "transitions": transitions,
        "analysis_seed": args.analysis_seed,
        "probes_per_task": len(manifest[0]),
    }
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
