#!/usr/bin/env python3
"""Extract deterministic intermediate features across continual checkpoints.

The output focuses on feature retention and consecutive phase drift.  It avoids
dimensionality-reduction plots: every comparison uses the same fixed probes and
linear CKA on pooled feature matrices.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from analyze_demo_gradients import (
    iter_batches,
    load_model,
    make_dataset,
    materialize_samples,
    read_manifest,
    set_batch_rng,
    write_rows,
)
from gr00t.model.transforms import DefaultDataCollator


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
        "--forgetting-csv",
        type=Path,
        default=Path(
            "outputs/analysis/libero_spatial/none/forgetting_metrics_seed42/"
            "combined_forgetting_metrics.csv"
        ),
    )
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/libero_spatial/none/phase_features_seed42"),
    )
    return parser.parse_args()


def checkpoint_path(args: argparse.Namespace, task: int) -> Path:
    return args.experiment_root / f"task_{task}" / f"checkpoint-{args.checkpoint_step}"


def tensor_from_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return tensor_from_output(output[0])
    if isinstance(output, Mapping) or hasattr(output, "data"):
        output = output.data if hasattr(output, "data") else output
        for key in ("backbone_features", "last_hidden_state", "hidden_states"):
            if key in output:
                return tensor_from_output(output[key])
    raise TypeError(f"Unsupported hook output: {type(output)}")


def pool_feature(name: str, value: torch.Tensor, action_horizon: int) -> torch.Tensor:
    value = value.detach().float()
    if value.ndim == 2:
        return value
    if value.ndim != 3:
        return value.flatten(1)
    if name in {"dit_early", "dit_middle", "dit_late", "action_decoder"}:
        value = value[:, -action_horizon:]
    if name == "action_decoder":
        return value.flatten(1)
    return value.mean(dim=1)


def collect_features(
    model: torch.nn.Module,
    samples: list[dict],
    collator: DefaultDataCollator,
    batch_size: int,
    seed: int,
    device_type: str,
) -> dict[str, np.ndarray]:
    captured: dict[str, torch.Tensor] = {}
    batches: dict[str, list[torch.Tensor]] = {name: [] for name in FEATURES}

    def hook(name):
        def capture(_module, _inputs, output):
            captured[name] = tensor_from_output(output)
        return capture

    blocks = model.action_head.model.transformer_blocks
    modules = {
        "backbone": model.backbone,
        "vl_projector": model.action_head.vl_self_attention,
        "state_encoder": model.action_head.state_encoder,
        "action_encoder": model.action_head.action_encoder,
        "dit_early": blocks[0],
        "dit_middle": blocks[len(blocks) // 2],
        "dit_late": blocks[-1],
        "action_decoder": model.action_head.action_decoder,
    }
    handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
    try:
        with torch.no_grad():
            for batch_index, sample_batch in enumerate(iter_batches(samples, batch_size)):
                set_batch_rng(seed, batch_index)
                batch = collator(sample_batch)
                captured.clear()
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    model(batch)
                missing = set(FEATURES) - set(captured)
                if missing:
                    raise RuntimeError(f"Feature hooks did not fire: {sorted(missing)}")
                for name in FEATURES:
                    batches[name].append(
                        pool_feature(name, captured[name], model.action_horizon).cpu()
                    )
                del batch
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(values).numpy() for name, values in batches.items()}


def centered_gram(features: np.ndarray) -> np.ndarray:
    values = features.astype(np.float64, copy=False)
    values = values - values.mean(axis=0, keepdims=True)
    gram = values @ values.T
    gram -= gram.mean(axis=0, keepdims=True)
    gram -= gram.mean(axis=1, keepdims=True)
    gram += gram.mean()
    return gram


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_gram = centered_gram(left)
    right_gram = centered_gram(right)
    numerator = float(np.sum(left_gram * right_gram))
    denominator = float(
        np.sqrt(np.sum(left_gram * left_gram) * np.sum(right_gram * right_gram))
    )
    return numerator / denominator if denominator > 0 else float("nan")


def normalized_displacement(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.linalg.norm(right.astype(np.float64) - left.astype(np.float64)))
    denominator = float(np.linalg.norm(left.astype(np.float64)))
    return numerator / denominator if denominator > 0 else float("nan")


def read_behavior(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    result = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            result[(int(row["transition"]), int(row["old_task"]))] = {
                "old_loss_change": float(row["old_loss_change"]),
                "success_change": float(row["success_change"]),
                "forgetting_amount": float(row["forgetting_amount"]),
            }
    return result


def main() -> None:
    args = parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if checkpoints != list(range(min(checkpoints), max(checkpoints) + 1)):
        raise ValueError("Checkpoints must form a contiguous range")
    if min(checkpoints) != 0:
        raise ValueError("Feature retention requires acquisition checkpoints starting at 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    bank_dir = args.output / "feature_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    manifest = read_manifest(args.probe_manifest)
    collator = DefaultDataCollator()
    device_type = torch.device(args.device).type
    sample_cache: dict[int, list[dict]] = {}

    def samples_for(task: int) -> list[dict]:
        if task not in sample_cache:
            rows = manifest[task]
            dataset = make_dataset(rows[0])
            sample_cache[task] = materialize_samples(dataset, rows)
            del dataset
        return sample_cache[task]

    feature_bank: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for checkpoint in checkpoints:
        print(f"[checkpoint {checkpoint}] loading model", flush=True)
        model = load_model(checkpoint_path(args, checkpoint), args.device)
        for task in range(checkpoint + 1):
            bank_path = bank_dir / f"checkpoint_{checkpoint}_task_{task}.npz"
            if bank_path.exists():
                print(f"  task {task}: reuse {bank_path.name}", flush=True)
                with np.load(bank_path) as stored:
                    feature_bank[(checkpoint, task)] = {
                        name: stored[name] for name in FEATURES
                    }
                continue
            print(f"  task {task}: extracting {len(manifest[task])} fixed probes", flush=True)
            features = collect_features(
                model,
                samples_for(task),
                collator,
                args.batch_size,
                args.analysis_seed,
                device_type,
            )
            np.savez_compressed(bank_path, **features)
            feature_bank[(checkpoint, task)] = features
        del model
        gc.collect()
        torch.cuda.empty_cache()

    behavior = read_behavior(args.forgetting_csv)
    transition_rows = []
    retention_rows = []
    max_checkpoint = max(checkpoints)
    for task in range(max_checkpoint + 1):
        reference = feature_bank[(task, task)]
        for checkpoint in range(task, max_checkpoint + 1):
            current = feature_bank[(checkpoint, task)]
            for feature in FEATURES:
                retention_rows.append(
                    {
                        "reference_checkpoint": task,
                        "checkpoint": checkpoint,
                        "task": task,
                        "feature": feature,
                        "retention_cka": linear_cka(reference[feature], current[feature]),
                        "cumulative_normalized_displacement": normalized_displacement(
                            reference[feature], current[feature]
                        ),
                    }
                )
        for checkpoint in range(task + 1, max_checkpoint + 1):
            previous = feature_bank[(checkpoint - 1, task)]
            current = feature_bank[(checkpoint, task)]
            for feature in FEATURES:
                row = {
                    "transition": checkpoint,
                    "old_task": task,
                    "feature": feature,
                    "consecutive_cka": linear_cka(previous[feature], current[feature]),
                    "feature_drift_one_minus_cka": 1.0
                    - linear_cka(previous[feature], current[feature]),
                    "consecutive_normalized_displacement": normalized_displacement(
                        previous[feature], current[feature]
                    ),
                    **behavior[(checkpoint, task)],
                }
                transition_rows.append(row)

    write_rows(args.output / "feature_transition_metrics.csv", transition_rows)
    write_rows(args.output / "feature_retention_metrics.csv", retention_rows)
    metadata = {
        "features": FEATURES,
        "analysis_seed": args.analysis_seed,
        "num_fixed_probes_per_task": len(manifest[0]),
        "comparison": "linear CKA and normalized activation displacement",
        "dit_pooling": "mean over final action_horizon tokens",
        "decoder_pooling": "flattened action-token predicted velocity",
        "determinism": "same fixed probes, Gaussian noise, and flow timestep per checkpoint",
    }
    (args.output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
