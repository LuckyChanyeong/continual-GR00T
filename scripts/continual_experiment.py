"""Prepare/check the experiment without data; execution is always explicit."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".config/libero"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from gr00t.experiment.continual import (manifests, write_json, read_json, load_manifest,
                                       check_datasets, validate_task_catalog, record_prefix)


def training_command(m, manifest_path):
    c = m["config"]
    return [sys.executable, str(ROOT / "scripts/gr00t_sequential_finetune.py"),
            "--dataset-path", *[t["dataset_path"] for t in m["tasks"]],
            "--output-dir", m["output_dir"], "--run-name", m["run_label"],
            "--base-model-path", c["base_model_path"], "--data-config", c["data_config"],
            "--embodiment-tag", c["embodiment_tag"], "--cl-type", m["method"],
            "--seed", str(m["seed"]), "--batch-size", str(c["batch_size"]),
            "--n-memories", str(c["n_memories"]), "--max-steps", str(c["max_steps"]),
            "--save-steps", str(c["max_steps"]), "--num-gpus", "1",
            "--no-tune-llm", "--no-tune-visual", "--task-order-manifest", str(manifest_path),
            "--evaluate-after-phase"]


def evaluation_commands(m, path, phase):
    if phase not in range(len(m["tasks"])):
        raise ValueError("Invalid phase")
    c = m["config"]
    checkpoint = Path(m["output_dir"]) / f"task_{phase}" / f"checkpoint-{c['max_steps']}"
    for position in range(phase + 1):
        task = m["tasks"][position]["canonical_task_id"]
        output = Path(m["output_dir"]) / f"task_{phase}" / "evaluation" / f"canonical_task_{task:02d}"
        command = [sys.executable, str(ROOT / "scripts/eval_continual_libero.py"),
                   "--args.model-path", str(checkpoint), "--args.task-order-manifest", str(path),
                   "--args.phase", str(phase), "--args.task-idx", str(position),
                   "--args.task-suite-name", m["suite"], "--args.data-config", c["data_config"],
                   "--args.embodiment-tag", c["embodiment_tag"], "--args.video-out-path", str(output),
                   "--args.num-trials-per-task", str(c["rollouts"]), "--args.seed", str(c["eval_seed"])]
        if c["save_video"]:
            command.append("--args.save-video")
        yield command


def validate_config(c):
    if c["data_config"] != "libero":
        raise ValueError("This rollout adapter requires data_config=libero; libero90 uses incompatible observation/action keys")
    for key in ("max_steps", "batch_size", "gradient_interval", "diagnostic_batches", "rollouts"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["max_steps"] % c["gradient_interval"]:
        raise ValueError("max_steps must end on a gradient recording point (drift reference)")
    if c["method"] not in ("ER", "none", "EWC"):
        raise ValueError("Unsupported CL method")
    for key in ("suites", "order_seeds", "training_seeds"):
        if not c[key] or len(c[key]) != len(set(c[key])):
            raise ValueError(f"{key} must be nonempty and unique")


def environment_record():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "flash-attn", "libero", "robosuite", "mujoco", "scikit-learn"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    import hashlib
    diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"])
    record = dict(python=sys.version, packages=packages,
                  git_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                  tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                  cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                  libero_source=read_json(ROOT / ".config/libero/source.json") if (ROOT / ".config/libero/source.json").is_file() else None)
    # Include uncommitted/new experiment code in provenance, not just tracked diff.
    files = [*ROOT.joinpath("scripts").glob("*.py"), *ROOT.joinpath("gr00t/experiment").glob("*.py")]
    record["code_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    try:
        record["gpu_0"] = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid,pci.bus_id,name,driver_version", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        record["gpu_0"] = None
    return record


def require_runtime():
    if not (Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").exists():
        raise RuntimeError("Run scripts/configure_libero.py and source .envrc first")
    from libero.libero import benchmark
    validate_task_catalog(benchmark)
    # Fail before expensive training if the simulator native runtime cannot load.
    from libero.libero.envs import OffScreenRenderEnv
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Assigned GPU 0 is unavailable; no fallback to another GPU")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Write manifests only; never load data/model")
    prepare.add_argument("--config", type=Path, default=ROOT / "configs/continual_analysis.json")
    prepare.add_argument("--attempt", default="01")
    doctor = sub.add_parser("doctor", help="Check packages/config; missing data is reported, not downloaded")
    doctor.add_argument("--config", type=Path, default=ROOT / "configs/continual_analysis.json")
    run = sub.add_parser("run", help="Print training commands; add --execute to run")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--index", type=Path)
    run.add_argument("--execute", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Print evaluation commands; add --execute to run")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--phase", type=int, required=True)
    evaluate.add_argument("--execute", action="store_true")
    analyze = sub.add_parser("analyze", help="Aggregate complete results and create plots")
    analyze.add_argument("--index", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--tsne", action="store_true", help="Optional offline gradient embedding")
    args = parser.parse_args()
    if args.command in ("prepare", "doctor"):
        c = read_json(args.config)
        validate_config(c)
        plans = list(manifests(c, ROOT, getattr(args, "attempt", "01")))
        if args.command == "prepare":
            for m in plans:
                path = Path(m["output_dir"]) / "task_order.json"
                if path.exists() and read_json(path) != m:
                    raise FileExistsError(f"Different configuration already exists: {path}")
            paths = []
            for m in plans:
                path = Path(m["output_dir"]) / "task_order.json"
                if not path.exists():
                    write_json(path, m)
                paths.append(str(path))
            index = ROOT / c["output_root"] / f"experiment_index__attempt{args.attempt}.json"
            value = dict(manifests=paths, config=c)
            if index.exists() and read_json(index) != value:
                raise FileExistsError(f"Index already exists with another configuration: {index}")
            write_json(index, value)
            print(f"Prepared {len(paths)} runs (no training or data downloads): {index}")
        else:
            report = environment_record()
            report["missing_dataset_paths"] = sorted({t["dataset_path"] for m in plans for t in m["tasks"]
                                                       if not (Path(t["dataset_path"]) / "meta/info.json").is_file()})
            report["planned_runs"] = len(plans)
            report["libero_config_exists"] = (Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").exists()
            if report["libero_config_exists"] and importlib.util.find_spec("libero"):
                from libero.libero import benchmark
                validate_task_catalog(benchmark)
                report["task_catalog_verified"] = True
            print(json.dumps(report, indent=2))
        return
    if args.command == "analyze":
        from gr00t.experiment.continual_analysis import analyze
        analyze(read_json(args.index)["manifests"], args.output, tsne=args.tsne)
        return
    paths = [args.manifest] if args.manifest else [Path(p) for p in read_json(args.index)["manifests"]]
    for path in paths:
        path = Path(path).resolve()
        m = load_manifest(path)
        validate_config(m["config"])
        commands = [training_command(m, path)] if args.command == "run" else list(evaluation_commands(m, path, args.phase))
        for command in commands:
            print(shlex.join(command), flush=True)
        if not args.execute:
            continue
        if args.command == "run" and any((Path(m["output_dir"]) / f"task_{i}").exists() for i in range(len(m["tasks"]))):
            raise FileExistsError("Run already has phase artifacts; use a new attempt (provenance was not overwritten)")
        check_datasets(m)
        require_runtime()
        if args.command == "run":
            import hashlib
            dataset_versions = []
            for task in m["tasks"]:
                folder = Path(task["dataset_path"]) / "meta"
                dataset_versions.append(dict(canonical_task_id=task["canonical_task_id"],
                    dataset_path=task["dataset_path"],
                    metadata_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(folder.iterdir()) if p.is_file()},
                    note="metadata only; not a checksum of every parquet/video file"))
            write_json(Path(m["output_dir"]) / "dataset_provenance.json", dataset_versions)
            write_json(Path(m["output_dir"]) / "environment.json", environment_record())
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
