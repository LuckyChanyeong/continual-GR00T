"""Data-free experiment protocol, stable task IDs, manifests and result files."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path


TASKS = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet", "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet", "push_the_plate_to_the_front_of_the_stove",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    ],
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def make_order(task_ids, seed):
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be nonempty and unique")
    order = list(task_ids)
    if seed is not None:
        random.Random(seed).shuffle(order)
    return order


def manifests(config, root, attempt="01"):
    root = Path(root).resolve()
    if not attempt or not all(c.isalnum() or c in "-_" for c in attempt):
        raise ValueError("attempt must contain only letters, digits, '-' or '_'")
    for suite in config["suites"]:
        if suite not in TASKS:
            raise ValueError(f"Unsupported suite: {suite}")
        if any(i not in range(6) for i in config["task_ids"]):
            raise ValueError("This protocol only supports canonical task IDs 0..5")
        for order_index, order_seed in enumerate(config["order_seeds"]):
            order = make_order(config["task_ids"], order_seed)
            for seed in config["training_seeds"]:
                label = f"{suite}__{config['method']}__order{order_index:02d}__oseed{order_seed if order_seed is not None else 'na'}__seed{seed}"
                output = (root / config["output_root"] / f"{label}__attempt{attempt}").resolve()
                entries = [dict(phase=p, canonical_task_id=i, task_name=TASKS[suite][i],
                                dataset_path=str((root / config["dataset_root"] / TASKS[suite][i]).resolve()))
                           for p, i in enumerate(order)]
                result = dict(schema_version=1, suite=suite, method=config["method"],
                              order_index=order_index, order_seed=order_seed, seed=seed,
                              task_order=order, tasks=entries, run_label=label, output_dir=str(output),
                              config=config)
                result["config_hash"] = fingerprint(result)
                yield result


def load_manifest(path):
    m = read_json(path)
    if fingerprint({k: v for k, v in m.items() if k != "config_hash"}) != m["config_hash"]:
        raise ValueError(f"Manifest hash mismatch: {path}")
    if [t["canonical_task_id"] for t in m["tasks"]] != m["task_order"]:
        raise ValueError("Task mapping mismatch")
    return m


def record_prefix(m, phase, step):
    task = m["tasks"][phase]["canonical_task_id"]
    return f"{m['run_label']}__phase{phase:02d}__task{task:02d}__step{step:05d}"


def check_datasets(m):
    missing = []
    for task in m["tasks"]:
        path = Path(task["dataset_path"])
        for filename in ("info.json", "modality.json", "episodes.jsonl", "tasks.jsonl", "stats.json"):
            if not (path / "meta" / filename).is_file():
                missing.append(str(path / "meta" / filename))
        if not (path / "data").is_dir():
            missing.append(str(path / "data"))
    if missing:
        raise FileNotFoundError("LeRobot datasets are not ready (no download was attempted):\n" + "\n".join(missing))


def validate_task_catalog(benchmark):
    for name, expected in TASKS.items():
        suite = benchmark.get_benchmark_dict()[name](task_order_index=0)
        actual = [suite.get_task(i).name for i in range(6)]
        if actual != expected:
            raise ValueError(f"LIBERO task catalog mismatch for {name}: {actual!r}")


def read_episodes(path, config_hash):
    """Reject stale/duplicate results, tolerate only an interrupted last line."""
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if row["config_hash"] != config_hash:
            raise ValueError(f"Evaluation config mismatch in {path}; use a new output directory")
        episode = row["episode_id"]
        if episode in rows:
            raise ValueError(f"Duplicate episode {episode} in {path}")
        rows[episode] = row
    return rows


def save_episodes(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for index in sorted(rows):
            stream.write(json.dumps(rows[index], allow_nan=False) + "\n")
    os.replace(temporary, path)
