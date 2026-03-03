#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import sys

# ---- paste your list here ----
RAW_DATASET_PATHS = [
    # "/path/to/your/dataset/task1/2024-04-25/demo.hdf5",
    # "/path/to/your/dataset/task2/2024-05-01/demo.hdf5",
]
# --------------------------------

def extract_task_name(hdf5_path: Path) -> str:
    """
    Expected structure: .../<task_name>/<date>/file.hdf5
    Example: .../TurnOnStove/2024-05-02/demo.hdf5  -> task_name = TurnOnStove
    """
    parts = hdf5_path.parts
    if len(parts) < 3:
        raise ValueError(f"Path too short to extract task name: {hdf5_path}")
    task = parts[-3]
    # Quick sanity checks:
    if task.endswith(".hdf5") or task.count(".") > 0 or len(task) == 0:
        raise ValueError(f"Could not parse task name from: {hdf5_path}")
    return task

def build_cmd(python_bin: str, converter_script: Path, src_hdf5: Path, out_root: Path) -> list[str]:
    task = extract_task_name(src_hdf5)
    output_path = out_root / task
    # Create directory lazily; converter might also handle this but it's safe
    output_path.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin, str(converter_script),
        "--src_hdf5_path", str(src_hdf5),
        "--output_path", str(output_path),
        "--task_name", task,
        "--meta_path", "meta_sample",
    ]
    
    # print out command
    # print(f"[DEBUG] Command to run: {' '.join(command)}")
    
    return command

def main():
    p = argparse.ArgumentParser(description="Batch LeRobot conversion runner.")
    p.add_argument("--python", default=sys.executable, help="Python executable to use (default: current interpreter).")
    p.add_argument("--converter", default="scripts/convert_hdf5_to_lerobot.py",
                   help="Path to convert_hdf5_to_lerobot.py")
    p.add_argument("--output-root", required=True,
                   help="Root directory for outputs (each task gets its own subfolder).")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = p.parse_args()

    converter_script = Path(args.converter)
    out_root = Path(args.output_root)

    failures = []
    for raw in RAW_DATASET_PATHS:
        src = Path(raw).expanduser().resolve()
        try:
            cmd = build_cmd(args.python, converter_script, src, out_root)
        except Exception as e:
            print(f"[SKIP] {raw}: {e}", file=sys.stderr)
            failures.append((raw, str(e)))
            continue

        print(" ".join(cmd))
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[FAILED] {raw}: exit code {e.returncode}", file=sys.stderr)
                failures.append((raw, f"returncode={e.returncode}"))

    if failures:
        print("\nSummary of failures:")
        for raw, msg in failures:
            print(f"- {raw}: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
