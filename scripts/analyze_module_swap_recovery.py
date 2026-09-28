#!/usr/bin/env python3
"""Measure old-task loss recovery after restoring one action-head module.

Two source checkpoints are evaluated:

* transition: restore module m from checkpoint k-1 into checkpoint k;
* acquisition: restore module m from checkpoint j, where old task j was learned.

Positive ``absolute_loss_recovery`` means that restoring the module lowers the
fixed-probe old-task flow loss.  This is an intervention, not merely a
correlation, although cross-module interactions mean it is not an additive
attribution.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from analyze_demo_gradients import (
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
    parser.add_argument(
        "--functional-metrics",
        type=Path,
        required=True,
        help="combined_forgetting_metrics.csv from analyze_forgetting_metrics.py",
    )
    parser.add_argument("--transitions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--modules", nargs="+", default=MODULES, choices=MODULES)
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/causal_first5_seed42/module_swap"
        ),
    )
    return parser.parse_args()


def checkpoint_path(args: argparse.Namespace, task: int) -> Path:
    return args.experiment_root / f"task_{task}" / f"checkpoint-{args.checkpoint_step}"


def read_weight_map(checkpoint: Path) -> dict[str, str]:
    with (checkpoint / "model.safetensors.index.json").open() as handle:
        return json.load(handle)["weight_map"]


def load_group_into_model(model, checkpoint: Path, module: str) -> None:
    mapping = read_weight_map(checkpoint)
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter_group(name) == module
    }
    missing = set(selected) - set(mapping)
    if missing:
        raise KeyError(f"Checkpoint {checkpoint} misses {len(missing)} parameters")
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(checkpoint / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(mapping[name] for name in selected))
        }
        with torch.no_grad():
            for name, parameter in selected.items():
                value = handles[mapping[name]].get_tensor(name)
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def mean_loss(model, samples, collator, batch_size, seed, device_type) -> float:
    total = 0.0
    total_samples = len(samples)
    with torch.no_grad():
        for batch_index, sample_batch in enumerate(iter_batches(samples, batch_size)):
            set_batch_rng(seed, batch_index)
            batch = collator(sample_batch)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                loss = model(batch)["loss"]
            total += float(loss.detach().float().item()) * len(sample_batch) / total_samples
            del batch, loss
    return total


def read_damage(path: Path):
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result[(int(row["transition"]), int(row["old_task"]))] = {
                "pre_loss": float(row["pre_old_loss"]),
                "post_loss": float(row["post_old_loss"]),
                "loss_change": float(row["old_loss_change"]),
                "success_change": float(row["success_change"]),
            }
    return result


def main() -> None:
    args = parse_args()
    transitions = sorted(set(args.transitions))
    modules = list(dict.fromkeys(args.modules))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    manifest = read_manifest(args.probe_manifest)
    damage = read_damage(args.functional_metrics)
    collator = DefaultDataCollator()
    device_type = torch.device(args.device).type
    rows = []

    for transition in transitions:
        post_checkpoint = checkpoint_path(args, transition)
        print(f"[transition {transition}] loading post-transition checkpoint", flush=True)
        model = load_model(post_checkpoint, args.device)
        sample_cache = {}

        def samples_for(task: int):
            if task not in sample_cache:
                task_rows = manifest[task]
                dataset = make_dataset(task_rows[0])
                sample_cache[task] = materialize_samples(dataset, task_rows)
                del dataset
            return sample_cache[task]

        baseline = {}
        for old_task in range(transition):
            baseline[old_task] = mean_loss(
                model,
                samples_for(old_task),
                collator,
                args.batch_size,
                args.analysis_seed,
                device_type,
            )
            expected = damage[(transition, old_task)]["post_loss"]
            if abs(baseline[old_task] - expected) > 2e-4:
                raise ValueError(
                    f"Baseline mismatch at transition={transition}, task={old_task}: "
                    f"{baseline[old_task]} vs {expected}"
                )

        transition_cache = {}
        for module in modules:
            print(f"  transition swap: {module}", flush=True)
            load_group_into_model(model, checkpoint_path(args, transition - 1), module)
            for old_task in range(transition):
                transition_cache[(old_task, module)] = mean_loss(
                    model,
                    samples_for(old_task),
                    collator,
                    args.batch_size,
                    args.analysis_seed,
                    device_type,
                )
            load_group_into_model(model, post_checkpoint, module)

        for old_task in range(transition):
            for source_mode, source_task in (
                ("transition", transition - 1),
                ("acquisition", old_task),
            ):
                for module in modules:
                    if source_mode == "transition" or source_task == transition - 1:
                        swapped_loss = transition_cache[(old_task, module)]
                    else:
                        print(
                            f"  acquisition swap: old task {old_task}, {module}",
                            flush=True,
                        )
                        load_group_into_model(
                            model, checkpoint_path(args, source_task), module
                        )
                        swapped_loss = mean_loss(
                            model,
                            samples_for(old_task),
                            collator,
                            args.batch_size,
                            args.analysis_seed,
                            device_type,
                        )
                        load_group_into_model(model, post_checkpoint, module)

                    baseline_loss = baseline[old_task]
                    loss_change = damage[(transition, old_task)]["loss_change"]
                    recovery = baseline_loss - swapped_loss
                    rows.append(
                        {
                            "analysis_seed": args.analysis_seed,
                            "transition": transition,
                            "old_task": old_task,
                            "source_mode": source_mode,
                            "source_checkpoint": source_task,
                            "module": module,
                            "num_probes": len(samples_for(old_task)),
                            "pre_transition_loss": damage[(transition, old_task)]["pre_loss"],
                            "baseline_post_loss": baseline_loss,
                            "transition_loss_change": loss_change,
                            "swapped_loss": swapped_loss,
                            "absolute_loss_recovery": recovery,
                            "damage_recovery_fraction": (
                                recovery / loss_change if loss_change > 0 else float("nan")
                            ),
                            "rollout_success_change": damage[(transition, old_task)]["success_change"],
                        }
                    )
                    write_rows(args.output / "module_swap_recovery.csv", rows)

        del model, sample_cache
        gc.collect()
        torch.cuda.empty_cache()

    metadata = {
        "transitions": transitions,
        "modules": modules,
        "source_modes": {
            "transition": "restore module from checkpoint k-1",
            "acquisition": "restore module from checkpoint j where old task j was acquired",
        },
        "positive_recovery": "restored module lowers old-task fixed-probe flow loss",
        "analysis_seed": args.analysis_seed,
        "probes_per_task": len(manifest[0]),
    }
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
