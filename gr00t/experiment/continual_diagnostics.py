"""Read-only AdamW direction diagnostics; no additional optimizer steps."""
from __future__ import annotations

from contextlib import contextmanager
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from gr00t.experiment.continual import fingerprint, record_prefix, write_json


@contextmanager
def isolated_diagnostic(model):
    cpu_rng, py_rng, np_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(m, m.training) for m in model.modules()]
    gradients = [(p, p.grad) for p in model.parameters()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for module, training in modes:
            module.training = training
        for parameter, grad in gradients:
            parameter.grad = grad
        torch.set_rng_state(cpu_rng)
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def seed_diagnostic(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def unwrap_optimizer(optimizer):
    while hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("Continual diagnostics currently support torch AdamW only")
    return optimizer


def direction(parameter, group, state, *, prospective=False):
    """u excludes LR/decay. Actual uses updated moments; prospective never mutates S."""
    if group.get("amsgrad", False) or group.get("maximize", False):
        raise ValueError("AMSGrad/maximize are not supported by this diagnostic")
    if parameter.grad is None:
        return torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
    beta1, beta2 = group["betas"]
    step = int(state.get("step", 0))
    if prospective:
        gradient = parameter.grad.detach()
        m = state["exp_avg"].detach().clone() if "exp_avg" in state else torch.zeros_like(parameter)
        v = state["exp_avg_sq"].detach().clone() if "exp_avg_sq" in state else torch.zeros_like(parameter)
        # Match AdamW moment arithmetic in the optimizer state dtype, including BF16 rounding.
        m.lerp_(gradient, 1 - beta1)
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        m, v = m.float(), v.float()
        step += 1
    else:
        if step < 1:
            raise ValueError("Cannot record an actual direction before an AdamW update")
        m, v = state["exp_avg"].detach().float(), state["exp_avg_sq"].detach().float()
    u = (m / (1 - beta1**step)) / ((v / (1 - beta2**step)).sqrt() + group["eps"])
    return u.detach().cpu()


def optimizer_directions(model, optimizer, *, prospective=False):
    optimizer = unwrap_optimizer(optimizer)
    groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    result, active = {}, []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        result[name] = direction(p, groups[id(p)], optimizer.state.get(p, {}), prospective=prospective)
        if p.grad is not None:
            active.append(name)
    return result, active


def module_name(name):
    parts = name.split(".")
    return ".".join(parts[:2]) if parts[0] == "action_head" else parts[0]


def vector_metrics(a, b):
    if a.keys() != b.keys() or any(a[n].shape != b[n].shape for n in a):
        raise ValueError("Gradient coordinates do not match")
    aa = bb = ab = dd = 0.0
    for n in a:
        # Reductions in double, one parameter tensor at a time (no giant concatenation).
        x, y = a[n].double(), b[n].double()
        aa += x.square().sum().item()
        bb += y.square().sum().item()
        ab += (x * y).sum().item()
        dd += (x - y).square().sum().item()
    denom = math.sqrt(aa * bb)
    cosine = max(-1.0, min(1.0, ab / denom)) if denom > 1e-20 else None
    return dict(cosine=cosine, direction_drift=None if cosine is None else 1 - cosine,
                relative_drift=math.sqrt(dd) / (math.sqrt(aa) + 1e-12),
                reference_norm=math.sqrt(aa), current_norm=math.sqrt(bb))


def comparisons(a, b):
    metrics = {"all": vector_metrics(a, b)}
    for module in sorted({module_name(n) for n in a}):
        metrics[module] = vector_metrics({n: v for n, v in a.items() if module_name(n) == module},
                                         {n: v for n, v in b.items() if module_name(n) == module})
    return metrics


def save_tensor_record(path, directions, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bytes_needed = sum(x.numel() * x.element_size() for x in directions.values())
    if shutil.disk_usage(path.parent).free < bytes_needed + 1024**3:
        raise OSError(f"Insufficient disk space for gradient/Fisher record: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({"metadata": metadata, "directions": directions}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ContinualDiagnostics(TrainerCallback):
    def __init__(self, manifest, phase, trainer, batch_provider):
        self.manifest, self.phase, self.trainer = manifest, phase, trainer
        self.batch_provider = batch_provider
        self.interval = manifest["config"]["gradient_interval"]
        self.folder = Path(manifest["output_dir"]) / f"task_{phase}" / "gradients"
        self.model = trainer.model
        self.pending = False
        self.coordinates = None

    def on_train_begin(self, args, state, control, **kwargs):
        names = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not names or any(not n.startswith("action_head.") for n, p in names):
            raise ValueError("Analysis requires frozen VLM and trainable action_head only")
        self.coordinates = [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype),
                             "numel": p.numel(), "module": module_name(n)} for n, p in names]
        self.coordinate_hash = fingerprint(self.coordinates)
        destination = Path(self.manifest["output_dir"]) / "parameter_manifest.json"
        if destination.exists():
            import json
            if json.loads(destination.read_text())["hash"] != self.coordinate_hash:
                raise ValueError("Trainable parameter coordinates changed between phases")
        else:
            write_json(destination, {"hash": self.coordinate_hash, "parameters": self.coordinates,
                                     "dimension": sum(p.numel() for n, p in names),
                                     "float32_bytes_per_vector": 4 * sum(p.numel() for n, p in names)})
        unwrap_optimizer(self.trainer.optimizer)
        # Conservative lower bound: raw directions/Fisher only, excludes model/optimizer checkpoints.
        pbytes = 4 * sum(p.numel() for n, p in names)
        ticks = self.manifest["config"]["max_steps"] // self.interval
        batches = self.manifest["config"].get("diagnostic_batches", 1)
        remaining = sum(ticks * (1 + batches * (1 if phase == 0 else 2)) + 1
                        for phase in range(self.phase, len(self.manifest["tasks"])))
        estimate = pbytes * remaining
        free = shutil.disk_usage(destination.parent).free
        write_json(destination.parent / f"storage_estimate__phase{self.phase:02d}.json",
                   dict(dimension=pbytes // 4, bytes_per_vector=pbytes,
                        remaining_gradient_fisher_bytes_lower_bound=estimate, available_bytes=free,
                        excludes="model/optimizer checkpoints, data, temporary files, other runs"))
        if estimate + 1024**3 > free:
            raise OSError("Full-vector gradient/Fisher recording exceeds free disk space. "
                          "Inspect storage_estimate__phase*.json; choose larger output storage or "
                          "a separate smaller pilot configuration. No silent compression is applied.")

    def metadata(self, step, kind, **extra):
        return dict(schema_version=1, config_hash=self.manifest["config_hash"],
                    parameter_hash=self.coordinate_hash, suite=self.manifest["suite"],
                    order_seed=self.manifest["order_seed"], seed=self.manifest["seed"],
                    phase=self.phase, task=self.manifest["task_order"][self.phase],
                    step=step, kind=kind, optimizer_reset_per_phase=True, **extra)

    def on_optimizer_step(self, args, state, control, **kwargs):
        step = state.global_step + 1
        if step % self.interval:
            return
        if self.trainer.accelerator.optimizer_step_was_skipped:
            write_json(self.folder / f"{record_prefix(self.manifest,self.phase,step)}__skipped.json",
                       self.metadata(step, "skipped_update"))
            return
        optimizer = unwrap_optimizer(self.trainer.optimizer)
        values, active = optimizer_directions(self.model, optimizer)
        group_metadata = [{k: g[k] for k in ("lr", "weight_decay", "betas", "eps")} for g in optimizer.param_groups]
        save_tensor_record(self.folder / f"{record_prefix(self.manifest,self.phase,step)}__train_actual.pt",
                           values, self.metadata(step, "train_actual", active_parameters=active,
                           groups=group_metadata, model_timing="pre_step", moment_timing="post_step"))
        self.pending = True

    def on_step_end(self, args, state, control, **kwargs):
        if not self.pending:
            return
        self.pending = False
        step = state.global_step
        task_ids = self.manifest["task_order"][max(0, self.phase - 1):self.phase + 1]
        metrics = []
        for batch_index in range(self.manifest["config"].get("diagnostic_batches", 1)):
            directions = {}
            for task in task_ids:
                with isolated_diagnostic(self.model):
                    batch, seed, indices = self.batch_provider(task, batch_index)
                    seed_diagnostic(seed)
                    inputs = self.trainer._prepare_inputs(batch)
                    with self.trainer.compute_loss_context_manager():
                        loss = self.model(inputs)["loss"]  # no replay mixture/EWC penalty
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm) if args.max_grad_norm and args.max_grad_norm > 0 else None
                    values, active = optimizer_directions(self.model, self.trainer.optimizer, prospective=True)
                    metadata = self.metadata(step, "task_diagnostic", evaltask=task, batch_index=batch_index,
                                             analysis_seed=seed, sample_indices=indices, active_parameters=active,
                                             model_timing="post_step", optimizer_timing="post_step_prospective_next",
                                             raw_gradient_norm=None if norm is None else float(norm))
                    path = self.folder / f"{record_prefix(self.manifest,self.phase,step)}__evaltask{task:02d}__batch{batch_index:02d}__task_diagnostic.pt"
                    save_tensor_record(path, values, metadata)
                    directions[task] = values
            if len(task_ids) == 2:
                old, current = task_ids
                metrics.append(dict(kind="conflict", old_task=old, current_task=current,
                                    batch_index=batch_index, metrics=comparisons(directions[old], directions[current])))
                previous = Path(self.manifest["output_dir"]) / f"task_{self.phase-1}" / "gradients"
                reference_step = self.manifest["config"]["max_steps"]
                ref = previous / f"{record_prefix(self.manifest,self.phase-1,reference_step)}__evaltask{old:02d}__batch{batch_index:02d}__task_diagnostic.pt"
                if not ref.exists():
                    raise FileNotFoundError(f"Previous phase drift reference is missing: {ref}")
                prior = torch.load(ref, map_location="cpu", weights_only=True)
                if prior["metadata"]["parameter_hash"] != self.coordinate_hash:
                    raise ValueError("Drift reference parameter map changed")
                metrics.append(dict(kind="same_task_drift", task=old, batch_index=batch_index,
                                    reference_step=reference_step,
                                    metrics=comparisons(prior["directions"], directions[old])))
        write_json(self.folder / f"{record_prefix(self.manifest,self.phase,step)}__metrics.json",
                   self.metadata(step, "diagnostic_metrics", records=metrics))
