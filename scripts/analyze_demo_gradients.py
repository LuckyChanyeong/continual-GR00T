#!/usr/bin/env python3
"""Measure module-wise flow-matching gradient interference on fixed demos.

For transition k, checkpoint k-1 is evaluated on incoming task k and all old
tasks j < k. The incoming gradient is retained on CPU only long enough to compute
exact dot products with the old-task gradients, then discarded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import DefaultDataCollator, GR00TTransform


GROUP_ORDER = [
    "state_encoder",
    "action_encoder",
    "vl_projector",
    "future_tokens",
    "dit",
    "action_decoder",
    "all_action_head",
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
    parser.add_argument("--transitions", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/libero_spatial/none/demo_gradients_seed42"),
    )
    return parser.parse_args()


def read_manifest(path: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped[int(row["task_id"])].append(row)
    return dict(grouped)


def parameter_group(name: str) -> str | None:
    if not name.startswith("action_head."):
        return None
    component = name.split(".", 2)[1]
    if component == "state_encoder":
        return "state_encoder"
    if component in {"action_encoder", "position_embedding"}:
        return "action_encoder"
    if component in {"vlln", "vl_self_attention"}:
        return "vl_projector"
    if component == "future_tokens":
        return "future_tokens"
    if component == "model":
        return "dit"
    if component == "action_decoder":
        return "action_decoder"
    raise ValueError(f"Unclassified trainable action-head parameter: {name}")


def iter_batches(items: list[dict[str, str]], batch_size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_dataset(row: dict[str, str]) -> LeRobotSingleDataset:
    data_config = DATA_CONFIG_MAP["libero"]
    transforms = data_config.transform()
    # Freeze crop/color preprocessing while retaining supervised action targets.
    # A composed eval() disables stochastic VideoCrop/VideoColorJitter, but the
    # final GR00TTransform must remain in training mode because it only emits
    # ``action`` and ``action_mask`` in that mode.
    transforms.eval()
    action_target_transforms = [
        transform for transform in transforms.transforms if isinstance(transform, GR00TTransform)
    ]
    if len(action_target_transforms) != 1:
        raise RuntimeError(
            f"Expected exactly one GR00TTransform, found {len(action_target_transforms)}"
        )
    action_target_transforms[0].train()
    return LeRobotSingleDataset(
        dataset_path=row["dataset_path"],
        modality_configs=data_config.modality_config(),
        transforms=transforms,
        embodiment_tag=EmbodimentTag("libero"),
        video_backend="decord",
    )


def materialize_samples(
    dataset: LeRobotSingleDataset, rows: list[dict[str, str]]
) -> list[dict]:
    samples = []
    for row in rows:
        raw = dataset.get_step_data(int(row["episode_id"]), int(row["base_index"]))
        samples.append(dataset.transforms(raw))
    return samples


def set_batch_rng(seed: int, batch_index: int) -> None:
    batch_seed = seed + 1_000_003 * batch_index
    random.seed(batch_seed)
    torch.manual_seed(batch_seed)
    np.random.seed(batch_seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(batch_seed)


def group_squared_norms(model: torch.nn.Module) -> dict[str, float]:
    squared = defaultdict(float)
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if group is None or parameter.grad is None:
            continue
        value = float(torch.linalg.vector_norm(parameter.grad.detach()).item())
        squared[group] += value * value
        squared["all_action_head"] += value * value
    return {group: math.sqrt(squared[group]) for group in GROUP_ORDER}


def capture_gradients_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    captured = {}
    for name, parameter in model.named_parameters():
        if parameter_group(name) is not None and parameter.grad is not None:
            captured[name] = parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
    return captured


def compare_with_reference(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
    reference_norms: dict[str, float],
) -> dict[str, dict[str, float]]:
    dots = defaultdict(float)
    current_squared = defaultdict(float)
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if group is None or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        ref = reference[name].to(device=grad.device, dtype=grad.dtype)
        dot = float(torch.dot(grad.reshape(-1), ref.reshape(-1)).item())
        norm = float(torch.linalg.vector_norm(grad).item())
        dots[group] += dot
        dots["all_action_head"] += dot
        current_squared[group] += norm * norm
        current_squared["all_action_head"] += norm * norm
        del ref

    result = {}
    for group in GROUP_ORDER:
        current_norm = math.sqrt(current_squared[group])
        reference_norm = reference_norms[group]
        denominator = reference_norm * current_norm
        result[group] = {
            "dot": dots[group],
            "incoming_norm": reference_norm,
            "old_norm": current_norm,
            "cosine": dots[group] / denominator if denominator > 0 else float("nan"),
        }
    return result


def compute_mean_gradient(
    model: torch.nn.Module,
    samples: list[dict],
    collator: DefaultDataCollator,
    batch_size: int,
    seed: int,
    device_type: str,
) -> float:
    model.zero_grad(set_to_none=True)
    batches = list(iter_batches(samples, batch_size))
    total_loss = 0.0
    total_samples = len(samples)

    for batch_index, sample_batch in enumerate(batches):
        set_batch_rng(seed, batch_index)
        batch = collator(sample_batch)
        weight = len(sample_batch) / total_samples
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            loss = model(batch)["loss"]
        total_loss += float(loss.detach().float().item()) * weight
        (loss * weight).backward()
        del batch, loss
    return total_loss


def load_model(checkpoint: Path, device: str) -> GR00T_N1_5:
    model = GR00T_N1_5.from_pretrained(
        str(checkpoint),
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
    )
    model.backbone.requires_grad_(False)
    model.action_head.requires_grad_(True)
    model.eval()
    model.to(device)
    return model


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for transition in args.transitions:
        if not 1 <= transition <= 9:
            raise ValueError(f"Transition must be in [1, 9], got {transition}")

    manifest = read_manifest(args.probe_manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    collator = DefaultDataCollator()
    output_rows: list[dict[str, object]] = []
    task_cache: dict[int, list[dict]] = {}

    def samples_for(task_id: int) -> list[dict]:
        if task_id not in task_cache:
            rows = manifest[task_id]
            dataset = make_dataset(rows[0])
            task_cache[task_id] = materialize_samples(dataset, rows)
            del dataset
        return task_cache[task_id]

    for transition in sorted(set(args.transitions)):
        checkpoint_task = transition - 1
        checkpoint = (
            args.experiment_root
            / f"task_{checkpoint_task}"
            / f"checkpoint-{args.checkpoint_step}"
        )
        print(
            f"[transition {transition}] loading checkpoint task {checkpoint_task}: {checkpoint}",
            flush=True,
        )
        model = load_model(checkpoint, args.device)
        device_type = torch.device(args.device).type

        incoming_samples = samples_for(transition)
        print(
            f"[transition {transition}] incoming task {transition}: "
            f"{len(incoming_samples)} fixed samples",
            flush=True,
        )
        incoming_loss = compute_mean_gradient(
            model,
            incoming_samples,
            collator,
            args.batch_size,
            args.analysis_seed,
            device_type,
        )
        incoming_norms = group_squared_norms(model)
        incoming_gradient = capture_gradients_cpu(model)

        for old_task in range(transition):
            old_samples = samples_for(old_task)
            print(
                f"[transition {transition}] old task {old_task}: "
                f"{len(old_samples)} fixed samples",
                flush=True,
            )
            old_loss = compute_mean_gradient(
                model,
                old_samples,
                collator,
                args.batch_size,
                args.analysis_seed,
                device_type,
            )
            comparisons = compare_with_reference(model, incoming_gradient, incoming_norms)
            for group in GROUP_ORDER:
                metrics = comparisons[group]
                output_rows.append(
                    {
                        "analysis_seed": args.analysis_seed,
                        "checkpoint_task": checkpoint_task,
                        "incoming_task": transition,
                        "old_task": old_task,
                        "module": group,
                        "num_samples_per_task": len(old_samples),
                        "incoming_flow_loss": incoming_loss,
                        "old_flow_loss": old_loss,
                        **metrics,
                    }
                )

            write_rows(args.output / "gradient_interference.csv", output_rows)

        del incoming_gradient, model
        task_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "analysis_seed": args.analysis_seed,
        "transitions": sorted(set(args.transitions)),
        "checkpoint_step": args.checkpoint_step,
        "batch_size": args.batch_size,
        "probe_manifest": str(args.probe_manifest.resolve()),
        "experiment_root": str(args.experiment_root.resolve()),
        "gradient_definition": "mean fixed-demo flow-matching gradient",
        "dropout": "disabled via model.eval()",
        "data_transform": (
            "deterministic eval crop/color transforms; only GR00TTransform remains in "
            "training mode to emit action targets"
        ),
        "precision": "BF16 autocast; FP32 trainable parameters and accumulated gradients",
    }
    with (args.output / "analysis_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
