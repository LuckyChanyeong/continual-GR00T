"""Lightweight, exact optimizer-step capture for later gradient/update analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import save_file
from transformers import TrainerCallback

from gr00t.model.action_head.flow_matching_action_head import CategorySpecificLinear


class AdamWStepAuditCallback(TrainerCallback):
    """Capture action-head state immediately before selected optimizer steps.

    At each requested optimizer step, this saves:
      * the pre-step action-head state needed to recompute old-task gradients;
      * the exact parameter displacement produced by optimizer.step();
      * scalar norms, learning rates, and parameter-group metadata.

    The callback intentionally does not save gradients.  The effective AdamW
    direction is the negative of the saved displacement; when every parameter
    group uses the same positive learning rate, ``g_eff = -delta / lr``.
    """

    def __init__(
        self,
        output_dir: str | Path,
        steps: tuple[int, ...] = (1000, 5000, 7500),
        base_model_path: str | None = None,
        delta_dtype: Literal["float32", "bfloat16"] = "float32",
    ) -> None:
        unique_steps = tuple(sorted(set(int(step) for step in steps)))
        if not unique_steps or any(step <= 0 for step in unique_steps):
            raise ValueError(f"Audit steps must be positive, got {unique_steps}")
        if delta_dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported delta dtype: {delta_dtype}")
        self.output_dir = Path(output_dir)
        self.steps = frozenset(unique_steps)
        self.base_model_path = base_model_path
        self.delta_dtype = torch.float32 if delta_dtype == "float32" else torch.bfloat16
        self._before: dict[str, torch.Tensor] | None = None
        self._active_category_rows: dict[str, list[int]] = {}
        self._learning_rates: list[float] = []
        self._parameter_group: dict[str, int] = {}
        self._pending_step: int | None = None

    @staticmethod
    def _action_head_parameters(model) -> dict[str, torch.nn.Parameter]:
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if name.startswith("action_head.") and parameter.requires_grad
        }
        if not parameters:
            raise RuntimeError("Optimizer audit found no trainable action_head parameters")
        return parameters

    @staticmethod
    def _category_parameter_names(model) -> set[str]:
        result = set()
        for module_name, module in model.named_modules():
            if isinstance(module, CategorySpecificLinear):
                result.add(module_name + ".W")
                result.add(module_name + ".b")
        return result

    @staticmethod
    def _active_rows(parameter: torch.nn.Parameter) -> list[int]:
        if parameter.grad is None:
            return []
        gradient = parameter.grad.detach()
        if gradient.ndim == 0:
            return []
        active = torch.count_nonzero(gradient.reshape(gradient.shape[0], -1), dim=1) > 0
        return active.nonzero(as_tuple=False).flatten().cpu().tolist()

    @staticmethod
    def _slice(tensor: torch.Tensor, rows: list[int] | None) -> torch.Tensor:
        if rows is None:
            return tensor
        if len(rows) == 1:
            return tensor[rows[0]]
        index = torch.tensor(rows, dtype=torch.long)
        return tensor.index_select(0, index)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        step = int(state.global_step) + 1
        if step not in self.steps:
            return control
        if not state.is_world_process_zero:
            return control
        if int(getattr(args, "world_size", 1)) != 1:
            raise RuntimeError("AdamWStepAuditCallback currently supports single-process training only")
        if self._before is not None:
            raise RuntimeError("Optimizer audit already has a pending pre-step snapshot")

        model = kwargs["model"]
        optimizer = kwargs["optimizer"]
        parameters = self._action_head_parameters(model)
        category_names = self._category_parameter_names(model)

        group_by_id = {}
        self._learning_rates = []
        for group_index, group in enumerate(optimizer.param_groups):
            self._learning_rates.append(float(group["lr"]))
            for parameter in group["params"]:
                group_by_id[id(parameter)] = group_index

        missing = [name for name, parameter in parameters.items() if id(parameter) not in group_by_id]
        if missing:
            raise RuntimeError(f"Trainable parameters missing from optimizer: {missing[:3]}")

        self._parameter_group = {
            name: group_by_id[id(parameter)] for name, parameter in parameters.items()
        }
        self._active_category_rows = {}
        for name, parameter in parameters.items():
            if name in category_names:
                rows = self._active_rows(parameter)
                if not rows:
                    raise RuntimeError(f"No active embodiment row in gradient for {name} at step {step}")
                self._active_category_rows[name] = rows

        # FP32 is required here: subtracting a BF16 snapshot would erase small
        # but valid optimizer updates before the delta is formed.
        self._before = {
            name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            for name, parameter in parameters.items()
        }
        self._pending_step = step
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        step = int(state.global_step) + 1
        if self._pending_step != step:
            return control
        if not state.is_world_process_zero:
            return control
        assert self._before is not None

        model = kwargs["model"]
        parameters = self._action_head_parameters(model)
        pre_state = {}
        delta_state = {}
        delta_squared = 0.0
        parameter_count = 0
        nonzero_count = 0

        for name, parameter in parameters.items():
            before = self._before[name]
            after = parameter.detach().to(device="cpu", dtype=torch.float32)
            delta = (after - before).contiguous()
            rows = self._active_category_rows.get(name)
            before = self._slice(before, rows).contiguous()
            delta = self._slice(delta, rows).contiguous()

            pre_state[name] = before.to(dtype=parameter.dtype)
            delta_state[name] = delta.to(dtype=self.delta_dtype)
            parameter_count += delta.numel()
            nonzero_count += int(torch.count_nonzero(delta))
            delta_squared += float(torch.sum(delta * delta, dtype=torch.float64))

        step_dir = self.output_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=False)
        pre_tmp = step_dir / "pre_step_action_head.tmp"
        delta_tmp = step_dir / "actual_update.tmp"
        save_file(pre_state, str(pre_tmp))
        save_file(delta_state, str(delta_tmp))
        pre_tmp.replace(step_dir / "pre_step_action_head.safetensors")
        delta_tmp.replace(step_dir / "actual_update.safetensors")

        metadata = {
            "optimizer_step": step,
            "delta_definition": "theta_after_optimizer_step - theta_before_optimizer_step",
            "effective_gradient_definition": "-delta / learning_rate",
            "base_model_path": self.base_model_path,
            "pre_step_state_dtype": "original_parameter_dtype",
            "parameter_dtypes": {name: str(p.dtype).removeprefix("torch.") for name, p in parameters.items()},
            "pre_step_state_exact": True,
            "actual_update_dtype": str(self.delta_dtype).removeprefix("torch."),
            "learning_rates_by_optimizer_group": self._learning_rates,
            "parameter_group_by_name": self._parameter_group,
            "active_category_rows": self._active_category_rows,
            "stored_parameter_count": parameter_count,
            "nonzero_update_count": nonzero_count,
            "actual_update_norm": math.sqrt(delta_squared),
            "scope": "trainable action_head parameters; category-specific tensors store active rows only",
            "reconstruction": (
                "Load base_model_path, overlay pre_step_action_head tensors, inserting tensors listed "
                "in active_category_rows into their original leading-dimension rows."
            ),
        }
        metadata_tmp = step_dir / "metadata.json.tmp"
        metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
        metadata_tmp.replace(step_dir / "metadata.json")

        self._before = None
        self._active_category_rows = {}
        self._learning_rates = []
        self._parameter_group = {}
        self._pending_step = None
        return control
