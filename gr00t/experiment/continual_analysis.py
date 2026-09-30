"""Offline aggregation: missing runs remain missing, never failed episodes or zeroes."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from gr00t.experiment.continual import load_manifest, read_json, write_json


def csv_file(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def success_matrix(m):
    n = len(m["tasks"])
    matrix = np.full((n, n), np.nan)
    for phase in range(n):
        folder = Path(m["output_dir"]) / f"task_{phase}" / "evaluation"
        for path in folder.glob("canonical_task_*/*__metrics.json"):
            result = read_json(path)
            if result.get("manifest_hash") != m["config_hash"]:
                raise ValueError(f"Stale evaluation result: {path}")
            position = result["task_position"]
            if result["phase"] != phase or not 0 <= position <= phase:
                raise ValueError(f"Invalid evaluation coordinates: {path}")
            if result["canonical_task_id"] != m["task_order"][position]:
                raise ValueError(f"Task mapping mismatch: {path}")
            if (not result["complete"] or result["episodes"] != m["config"]["rollouts"]
                    or result["requested"] != m["config"]["rollouts"]):
                continue
            value = result["success_rate"]
            if value is None or not 0 <= value <= 1:
                raise ValueError(f"Invalid success rate: {path}")
            if np.isfinite(matrix[phase, position]):
                raise ValueError(f"Duplicate complete evaluation: {path}")
            matrix[phase, position] = value
    return matrix


def run_summary(matrix):
    n = len(matrix)
    expected = matrix[np.tril_indices(n)]
    if not np.isfinite(expected).all():
        return {"complete": False, "final_average_success": None, "final_forgetting": None}
    # Maximum is computed within each run, BEFORE averaging seeds.
    forgetting = [max(matrix[j:n-1, j]) - matrix[n-1, j] for j in range(n - 1)]
    return {"complete": True, "final_average_success": float(matrix[-1].mean()),
            "final_forgetting": float(np.mean(forgetting)) if forgetting else None}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def heatmap(matrix, labels, title, path, *, vmin=0, vmax=1):
    import seaborn as sns
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, mask=~np.isfinite(matrix), annot=True, fmt=".2f", vmin=vmin, vmax=vmax,
                xticklabels=labels, yticklabels=range(len(matrix)), ax=ax, cmap="viridis")
    ax.set(xlabel="Canonical evaluated task ID (learning order)", ylabel="Phase", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sketch(directions, dimensions=512, seed=0):
    """Shared linear CountSketch; visualization only, not quantitative conflict/drift."""
    result = np.zeros(dimensions, dtype=np.float64)
    offset = 0
    # Stable coordinates across all records, independent of serialization insertion order.
    for name in sorted(directions):
        values = directions[name].detach().cpu().float().numpy().reshape(-1)
        for start in range(0, len(values), 65536):
            block = values[start:start+65536]
            indices = np.arange(offset + start, offset + start + len(block), dtype=np.uint64)
            # SplitMix64 counter hash; same linear map for every input vector.
            hashed = indices + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
            hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            hashed ^= hashed >> np.uint64(31)
            bins = (hashed % np.uint64(dimensions)).astype(np.int64)
            signs = np.where((hashed >> np.uint64(63)) != 0, 1.0, -1.0)
            result += np.bincount(bins, weights=block * signs, minlength=dimensions)
        offset += len(values)
    return result


def gradient_embedding(plans, output):
    import torch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    projected, labels, coordinate_hash = [], [], None
    for m in plans:
        for path in sorted(Path(m["output_dir"]).glob("task_*/gradients/*__task_diagnostic.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            meta = record["metadata"]
            if meta["config_hash"] != m["config_hash"]:
                raise ValueError(f"Stale gradient record: {path}")
            current_hash = meta["parameter_hash"]
            coordinate_hash = coordinate_hash or current_hash
            if current_hash != coordinate_hash:
                raise ValueError("Cannot embed records with different parameter coordinates")
            vector = sketch(record["directions"])
            norm = np.linalg.norm(vector)
            if norm <= 1e-20:
                continue
            projected.append(vector / norm)
            labels.append(dict(suite=m["suite"], order_index=m["order_index"], seed=m["seed"],
                               phase=meta["phase"], task=meta["evaltask"], step=meta["step"], file=str(path)))
            del record
    settings = dict(input="task_diagnostic", method="shared CountSketch -> L2 normalization -> joint PCA -> joint t-SNE",
                    sketch_dimensions=512, sketch_seed=0, random_state=0, parameter_hash=coordinate_hash,
                    points=len(labels), interpretation="exploratory; metrics use original coordinates, not this projection")
    if len(labels) < 4:
        write_json(output / "tsne_settings.json", dict(settings, status="insufficient_points"))
        return
    data = np.stack(projected)
    components = min(50, len(data) - 1, data.shape[1])
    reduced = PCA(n_components=components, svd_solver="full").fit_transform(data)
    perplexity = min(30.0, (len(data) - 1) / 3)
    coordinates = TSNE(n_components=2, perplexity=perplexity, random_state=0,
                       init="random", learning_rate="auto").fit_transform(reduced)
    rows = [dict(label, x=float(x), y=float(y)) for label, (x, y) in zip(labels, coordinates)]
    csv_file(output / "tsne_points.csv", rows)
    write_json(output / "tsne_settings.json", dict(settings, status="complete",
               pca_components=components, perplexity=perplexity))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 8))
    for suite, task in sorted({(r["suite"], r["task"]) for r in rows}):
        points = np.array([[r["x"], r["y"]] for r in rows if r["suite"] == suite and r["task"] == task])
        ax.scatter(points[:, 0], points[:, 1], s=12, alpha=.6, label=f"{suite}/{task}")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1))
    ax.set_title("Joint embedding (exploratory; see CSV for phase/order/seed)")
    fig.tight_layout()
    fig.savefig(output / "tsne.png", dpi=160)
    plt.close(fig)


def analyze(paths, output, *, tsne=False):
    import torch
    from gr00t.experiment.continual_diagnostics import module_name
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plans = [load_manifest(path) for path in paths]
    groups, summaries, cells, diagnostics, fisher_rows = defaultdict(list), [], [], [], []
    for m in plans:
        tag = {k: m[k] for k in ("suite", "method", "order_index", "order_seed", "seed")}
        matrix = success_matrix(m)
        summary = run_summary(matrix)
        summaries.append(dict(tag, **summary, output_dir=m["output_dir"]))
        groups[(m["suite"], m["method"], m["order_index"], m["order_seed"])].append((m, matrix, summary))
        for phase, position in zip(*np.tril_indices(len(matrix))):
            value = matrix[phase, position]
            cells.append(dict(tag, phase=phase, canonical_task_id=m["task_order"][position],
                              success_rate=float(value) if np.isfinite(value) else None, complete=bool(np.isfinite(value))))
        for path in sorted(Path(m["output_dir"]).glob("task_*/gradients/*__metrics.json")):
            document = read_json(path)
            if document["config_hash"] != m["config_hash"]:
                raise ValueError(f"Stale diagnostic metrics: {path}")
            for record in document["records"]:
                for module, values in record["metrics"].items():
                    diagnostics.append(dict(tag, phase=document["phase"], step=document["step"], module=module,
                                            kind=record["kind"], batch_index=record["batch_index"], **values))
        for path in sorted(Path(m["output_dir"]).glob("task_*/fisher/*__ewc_fisher.pt")):
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["metadata"]["config_hash"] != m["config_hash"]:
                raise ValueError(f"Stale Fisher: {path}")
            totals = defaultdict(lambda: [0., 0])
            for name, value in record["directions"].items():
                totals[module_name(name)][0] += value.double().sum().item()
                totals[module_name(name)][1] += value.numel()
            for module, (total, count) in totals.items():
                fisher_rows.append(dict(tag, phase=record["metadata"]["phase"], module=module,
                                        fisher_sum=total, fisher_mean=total / count, parameters=count))
            del record
    csv_file(output / "per_run_summary.csv", summaries)
    csv_file(output / "success_cells.csv", cells)
    csv_file(output / "gradient_metrics.csv", diagnostics)
    csv_file(output / "fisher_by_module.csv", fisher_rows)
    aggregates = []
    for key, items in groups.items():
        first = items[0][0]
        expected = set(first["config"]["training_seeds"])
        seeds = [m["seed"] for m, _, _ in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seed/attempt in index; select exactly one attempt per condition")
        if any(m["task_order"] != first["task_order"] or m["config"] != first["config"] for m, _, _ in items):
            raise ValueError("Cannot aggregate seeds with different task order/configuration")
        complete = set(seeds) == expected and all(s["complete"] for _, _, s in items)
        row = dict(zip(("suite", "method", "order_index", "order_seed"), key),
                   complete=complete, expected_seeds=len(expected), completed_seeds=sum(s["complete"] for _, _, s in items))
        if complete:
            for metric in ("final_average_success", "final_forgetting"):
                values = [s[metric] for _, _, s in items]
                if all(v is not None for v in values):
                    row[metric + "_mean"], row[metric + "_std"] = mean_std(values)
            cube = np.stack([matrix for _, matrix, _ in items])
            mean, std = np.full(cube.shape[1:], np.nan), np.full(cube.shape[1:], np.nan)
            for phase, position in zip(*np.tril_indices(len(mean))):
                mean[phase, position], deviation = mean_std(cube[:, phase, position])
                std[phase, position] = deviation if deviation is not None else np.nan
            label = f"{key[0]}__{key[1]}__order{key[2]:02d}"
            heatmap(mean, first["task_order"], label + " seed mean", output / (label + "__success.png"))
            if len(seeds) > 1:
                heatmap(std, first["task_order"], label + " seed SD", output / (label + "__std.png"))
            plt = _pyplot()
            fig, ax = plt.subplots()
            for matrix in cube:
                ax.plot([np.mean(matrix[i, :i+1]) for i in range(len(matrix))], alpha=.5)
            ax.set(xlabel="Phase", ylabel="Average success on seen tasks", ylim=(0, 1), title=label)
            fig.tight_layout()
            fig.savefig(output / (label + "__curve.png"), dpi=160)
            plt.close(fig)
        aggregates.append(row)
    csv_file(output / "per_order_seed_mean_std.csv", aggregates)
    # Keep phase-local x coordinates; never join curves across optimizer resets.
    plt = _pyplot()
    for kind in ("conflict", "same_task_drift"):
        selected = [r for r in diagnostics if r["kind"] == kind and r["module"] == "all"]
        for suite in sorted({r["suite"] for r in selected}):
            fig, ax = plt.subplots()
            curves = defaultdict(list)
            for r in selected:
                if r["suite"] == suite:
                    curves[(r["order_index"], r["seed"], r["phase"], r["batch_index"])].append(r)
            for label, points in curves.items():
                points.sort(key=lambda r: r["step"])
                ax.plot([r["step"] for r in points], [r["cosine"] for r in points], alpha=.5, label=str(label))
            ax.set(xlabel="Phase-local optimizer step", ylabel="Cosine", ylim=(-1, 1), title=f"{suite}: {kind}")
            if len(curves) <= 12:
                ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(output / f"{suite}__{kind}.png", dpi=160)
            plt.close(fig)
    for suite in sorted({r["suite"] for r in fisher_rows}):
        selected = [r for r in fisher_rows if r["suite"] == suite]
        modules = sorted({r["module"] for r in selected})
        labels = sorted({(r["order_index"], r["seed"], r["phase"]) for r in selected})
        values = np.full((len(labels), len(modules)), np.nan)
        for r in selected:
            values[labels.index((r["order_index"], r["seed"], r["phase"])), modules.index(r["module"])] = r["fisher_mean"]
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(12, max(4, len(labels)*.2)))
        sns.heatmap(values, xticklabels=modules, yticklabels=[str(x) for x in labels], ax=ax)
        ax.set(title=f"{suite}: mean legacy Fisher per module", ylabel="(order, seed, phase)")
        fig.tight_layout()
        fig.savefig(output / f"{suite}__fisher.png", dpi=160)
        plt.close(fig)
    write_json(output / "analysis_status.json", dict(runs=len(plans), complete_runs=sum(s["complete"] for s in summaries),
               complete_order_groups=sum(r["complete"] for r in aggregates),
               missing_results="excluded, never converted to zero", seed_std_ddof=1))
    if tsne:
        gradient_embedding(plans, output)
    print(f"Analysis written to {output}; complete runs: {sum(s['complete'] for s in summaries)}/{len(plans)}")
