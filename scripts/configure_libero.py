"""Configure project-local LIBERO paths without downloading demonstrations."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def configure(root):
    root = Path(root).resolve()
    source = root / "third_party/LIBERO/libero/libero"
    if not (source / "bddl_files").is_dir():
        raise FileNotFoundError("Install the official LIBERO repository in third_party/LIBERO first")
    directory = root / ".config/libero"
    directory.mkdir(parents=True, exist_ok=True)
    dataset = root / "data/libero"
    dataset.mkdir(parents=True, exist_ok=True)
    config = {"benchmark_root": str(source), "bddl_files": str(source / "bddl_files"),
              "init_states": str(source / "init_files"), "assets": str(source / "assets"),
              "datasets": str(dataset)}
    # JSON is also valid YAML, avoiding an import that triggers LIBERO's prompt.
    (directory / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    commit = subprocess.check_output(["git", "-C", str(source.parents[1]), "rev-parse", "HEAD"], text=True).strip()
    (directory / "source.json").write_text(json.dumps({"repository": "https://github.com/Lifelong-Robot-Learning/LIBERO.git", "commit": commit}, indent=2) + "\n")
    print(f"Configured {directory}; no datasets or model weights downloaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    configure(parser.parse_args().root)
