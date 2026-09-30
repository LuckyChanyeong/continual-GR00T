"""CPU-only protocol tests: no datasets, model downloads or GPU allocation."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Trainer, TrainingArguments

from gr00t.experiment.continual import (fingerprint, load_manifest, make_order, manifests, read_episodes,
                                       save_episodes, write_json, record_prefix)
from gr00t.experiment.continual_diagnostics import (ContinualDiagnostics, direction,
    isolated_diagnostic, optimizer_directions, seed_diagnostic, vector_metrics)
from gr00t.experiment.continual_analysis import run_summary, success_matrix, sketch, analyze


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/continual_analysis.json").read_text())


def test_seed_grid(config, tmp_path):
    random.seed(41)
    state = random.getstate()
    plans = list(manifests(config, tmp_path))
    assert random.getstate() == state
    assert len(plans) == 36
    assert len({p["output_dir"] for p in plans}) == 36
    for suite in config["suites"]:
        members = [p for p in plans if p["suite"] == suite]
        assert members[0]["task_order"] == list(range(6))
        for order_index in range(4):
            subset = [p for p in members if p["order_index"] == order_index]
            assert {p["seed"] for p in subset} == {42, 43, 44}
            assert len({tuple(p["task_order"]) for p in subset}) == 1
    assert make_order(list(range(6)), 0) != list(range(6))


def test_hash_and_missing_data(config, tmp_path):
    from gr00t.experiment.continual import check_datasets
    m = next(manifests(config, tmp_path))
    path = tmp_path / "manifest.json"
    write_json(path, m)
    assert load_manifest(path) == m
    with pytest.raises(FileNotFoundError, match="no download"):
        check_datasets(m)
    m["seed"] += 1
    write_json(path, m)
    with pytest.raises(ValueError, match="hash"):
        load_manifest(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adamw_candidate_equals_committed_direction(dtype):
    p = torch.nn.Parameter(torch.tensor([.4, -.6, .2], dtype=dtype))
    optimizer = torch.optim.AdamW([p], lr=.03, betas=(.95, .999), weight_decay=.02)
    for step in range(4):
        p.grad = torch.tensor([.1 + step, -.2, .3], dtype=dtype)
        group = optimizer.param_groups[0]
        before_p = p.detach().clone()
        before_state = copy.deepcopy(optimizer.state.get(p, {}))
        candidate = direction(p, group, optimizer.state.get(p, {}), prospective=True)
        assert torch.equal(p, before_p)
        for key, value in before_state.items():
            assert torch.equal(optimizer.state[p][key], value)
        optimizer.step()
        actual = direction(p, group, optimizer.state[p])
        torch.testing.assert_close(candidate, actual, atol=2e-6, rtol=2e-6)
        if dtype == torch.float32:
            torch.testing.assert_close(p, before_p * (1 - group["lr"] * group["weight_decay"]) - group["lr"] * actual,
                                       atol=1e-6, rtol=1e-6)
    p.grad = None
    assert torch.count_nonzero(direction(p, group, optimizer.state[p])) == 0


def test_parameter_groups_and_unused_parameters():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW([{"params": [model.weight], "lr": .01},
                                   {"params": [model.bias], "lr": .02}])
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    values, active = optimizer_directions(model, optimizer)
    assert active == ["weight"]
    assert torch.count_nonzero(values["bias"]) == 0
    assert "step" not in optimizer.state[model.bias]
    assert vector_metrics(values, values)["cosine"] == pytest.approx(1.)
    assert vector_metrics({"a": torch.zeros(2)}, {"a": torch.ones(2)})["cosine"] is None
    with pytest.raises(ValueError):
        vector_metrics({"a": torch.ones(2)}, {"a": torch.ones(3)})


def test_isolation_restores_even_after_exception():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Dropout())
    model.train()
    model[1].eval()
    p = next(model.parameters())
    p.grad = torch.ones_like(p)
    original_grad = p.grad
    buffers = [b.clone() for b in model.buffers()]
    modes = [m.training for m in model.modules()]
    seed_diagnostic(23)
    states = torch.get_rng_state(), random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError):
        with isolated_diagnostic(model):
            seed_diagnostic(99)
            p.grad = torch.zeros_like(p)
            model[1].running_mean.add_(4)
            raise RuntimeError("test")
    assert p.grad is original_grad
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(a, b) for a, b in zip(buffers, model.buffers()))
    assert torch.equal(states[0], torch.get_rng_state())
    assert states[1] == random.getstate()
    assert np.array_equal(states[2][1], np.random.get_state()[1])


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3).requires_grad_(False)
        self.action_head = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.2))

    def forward(self, x=None):
        if isinstance(x, dict):
            x = x["x"]
        return {"loss": self.action_head(self.backbone(x)).square().mean()}


def test_real_trainer_callbacks_are_noninvasive(config, tmp_path):
    config = copy.deepcopy(config)
    config.update(max_steps=4, gradient_interval=2, task_ids=[0, 1], diagnostic_batches=1)
    m = next(manifests(config, tmp_path))
    seed_diagnostic(30)
    initial = Toy().state_dict()
    dataset = [{"x": torch.tensor([float(i), .1, -.3])} for i in range(8)]

    def train(enabled, phase):
        model = Toy()
        model.load_state_dict(initial)
        args = TrainingArguments(output_dir=str(tmp_path / f"trainer_{enabled}_{phase}"), max_steps=4,
              per_device_train_batch_size=2, report_to=[], use_cpu=True, save_strategy="no",
              disable_tqdm=True, learning_rate=.01, optim="adamw_torch", seed=42)
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        if enabled:
            def provider(task, batch_index):
                seed = task + 100
                seed_diagnostic(seed)
                return {"x": torch.randn(2, 3)}, seed, [0, 1]
            trainer.add_callback(ContinualDiagnostics(m, phase, trainer, provider))
        trainer.train()
        return copy.deepcopy(model.state_dict()), torch.get_rng_state()

    plain, plain_rng = train(False, 0)
    recorded, recorded_rng = train(True, 0)
    for name in plain:
        torch.testing.assert_close(plain[name], recorded[name], rtol=0, atol=0)
    assert torch.equal(plain_rng, recorded_rng)
    phase0 = Path(m["output_dir"]) / "task_0/gradients"
    assert len(list(phase0.glob("*__train_actual.pt"))) == 2
    assert len(list(phase0.glob("*__task_diagnostic.pt"))) == 2
    train(True, 1)
    phase1 = Path(m["output_dir"]) / "task_1/gradients"
    metrics = json.loads(next(phase1.glob("*__metrics.json")).read_text())
    assert {r["kind"] for r in metrics["records"]} == {"conflict", "same_task_drift"}


def test_episode_resume_validation(tmp_path):
    path = tmp_path / "episodes.jsonl"
    rows = {i: dict(episode_id=i, config_hash="same", status="complete", success=(i % 2 == 0)) for i in range(3)}
    save_episodes(path, rows)
    assert read_episodes(path, "same") == rows
    with path.open("a") as f:
        f.write('{"interrupted":')
    assert read_episodes(path, "same") == rows
    with pytest.raises(ValueError, match="mismatch"):
        read_episodes(path, "different")
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_episodes(path, "same")


def test_summary_and_shared_linear_projection():
    a = np.array([[.8, np.nan, np.nan], [.6, .9, np.nan], [.4, .8, .7]])
    assert run_summary(a)["final_average_success"] == pytest.approx(.6333333333)
    assert run_summary(a)["final_forgetting"] == pytest.approx(.25)
    a[1, 0] = np.nan
    assert not run_summary(a)["complete"]
    x, y = torch.arange(7.), torch.arange(7.) * -.3
    np.testing.assert_allclose(sketch({"a": x + y}), sketch({"a": x}) + sketch({"a": y}), atol=1e-6)


def test_analysis_requires_all_three_complete_seeds(config, tmp_path):
    config = copy.deepcopy(config)
    config.update(suites=["libero_spatial"], order_seeds=[None], task_ids=[0, 1])
    plans = list(manifests(config, tmp_path))
    paths = []
    for index, m in enumerate(plans):
        path = Path(m["output_dir"]) / "task_order.json"
        write_json(path, m)
        paths.append(path)
        for phase in range(2):
            for pos in range(phase + 1):
                target = Path(m["output_dir"]) / f"task_{phase}/evaluation/canonical_task_{pos:02d}/test__metrics.json"
                write_json(target, dict(manifest_hash=m["config_hash"], phase=phase, task_position=pos,
                           canonical_task_id=pos, complete=(index != 2), episodes=30 if index != 2 else 29,
                           requested=30, success_rate=.5 + .1*index))
    output = tmp_path / "analysis"
    analyze(paths, output)
    assert json.loads((output / "analysis_status.json").read_text())["complete_order_groups"] == 0
    assert np.isnan(success_matrix(plans[2])).all()
    for file in Path(plans[2]["output_dir"]).glob("task_*/evaluation/*/*__metrics.json"):
        value = json.loads(file.read_text())
        value.update(complete=True, episodes=30)
        write_json(file, value)
    analyze(paths, output)
    assert json.loads((output / "analysis_status.json").read_text())["complete_order_groups"] == 1
    assert list(output.glob("*__success.png"))


def test_eval_no_video_and_rollout_contract():
    from scripts.eval_continual_libero import Args, rollout, episode_seed
    args = Args(model_path="unused", num_steps_wait=0, replan_steps=1)
    assert args.num_trials_per_task == 30 and not args.save_video
    obs = dict(agentview_image=np.zeros((8, 8, 3), dtype=np.uint8),
               robot0_eye_in_hand_image=np.zeros((8, 8, 3), dtype=np.uint8),
               robot0_eef_pos=np.zeros(3), robot0_eef_quat=np.array([0., 0., 0., 1.]),
               robot0_gripper_qpos=np.zeros(2))
    class Environment:
        def reset(self): return obs
        def set_init_state(self, state): return obs
        def step(self, action):
            assert len(action) == 7
            return obs, 0., True, {}
    class Policy:
        def get_action(self, element):
            return {"action.eef_pos_delta": np.zeros((1, 3)), "action.eef_rot_delta": np.zeros((1, 3)),
                    "action.gripper_close": np.zeros((1, 1))}
    assert rollout(Policy(), Environment(), None, "test", args, 2) == (True, 1, None)
    assert episode_seed(7, 2, 9) == episode_seed(7, 2, 9)


def test_execution_is_opt_in(config, tmp_path):
    from scripts.continual_experiment import training_command, evaluation_commands
    m = next(manifests(config, tmp_path))
    command = training_command(m, tmp_path / "task_order.json")
    assert "--no-tune-llm" in command and "--no-tune-visual" in command
    assert command[command.index("--num-gpus") + 1] == "1"
    commands = list(evaluation_commands(m, tmp_path / "task_order.json", 5))
    assert len(commands) == 6 and all("--args.save-video" not in c for c in commands)



def test_legacy_fisher_formula_and_source():
    import ast
    import hashlib
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate
    source = (ROOT / "scripts/gr00t_sequential_finetune.py").read_text()
    section = source[source.index("def compute_fisher("):source.index("class EWCDualBrainTrainer")]
    # The original estimator body is intentionally kept intact, including batch=8.
    original_end = "class EWCDualBrainTrainer(DualBrainTrainer):\n"
    assert hashlib.sha256((section + original_end).encode()).hexdigest() == "34de644f961744930c99030c786a8b8efe3adb7cf58230de38971345cca104bc"
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "compute_fisher")
    namespace = dict(torch=torch, DataLoader=DataLoader, DefaultDataCollator=lambda: default_collate)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "legacy_fisher", "exec"), namespace)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_head = torch.nn.Linear(1, 1, bias=False)
        def forward(self, batch):
            return {"loss": self.action_head(batch["x"]).mean()}
    dataset = [{"x": torch.tensor([float(i)])} for i in range(9)]
    model = Model()
    seed_diagnostic(13)
    expected = torch.stack([batch["x"].mean().square() for batch in DataLoader(
        dataset, batch_size=8, shuffle=True, collate_fn=default_collate)]).mean()
    seed_diagnostic(13)
    with isolated_diagnostic(model):
        actual = namespace["compute_fisher"](model, dataset, "cpu", num_workers=0)
    assert actual["action_head.weight"].item() == pytest.approx(expected.item())


def test_tsne_pipeline_with_small_records(config, tmp_path):
    from gr00t.experiment.continual_analysis import gradient_embedding
    m = next(manifests(config, tmp_path))
    folder = Path(m["output_dir"]) / "task_0/gradients"
    folder.mkdir(parents=True)
    for step in range(6):
        generator = torch.Generator().manual_seed(step)
        torch.save({"metadata": dict(config_hash=m["config_hash"], parameter_hash="test_coordinates",
                    phase=0, evaltask=0, step=step),
                    "directions": {"action_head.a": torch.randn(12, generator=generator)}},
                   folder / f"{step}__task_diagnostic.pt")
    output = tmp_path / "tsne"
    output.mkdir()
    gradient_embedding([m], output)
    assert json.loads((output / "tsne_settings.json").read_text())["status"] == "complete"
    assert (output / "tsne.png").is_file()


def test_libero_schema_matches_rollout_and_rejects_libero90(config):
    import ast
    from scripts.continual_experiment import validate_config
    from scripts.eval_continual_libero import Args, eval_libero
    assert config["data_config"] == Args(model_path="unused").data_config == "libero"
    validate_config(config)
    with pytest.raises(ValueError, match="incompatible"):
        validate_config(dict(config, data_config="libero90"))
    with pytest.raises(ValueError, match="requires data_config"):
        eval_libero(Args(model_path="unused", data_config="libero90"))
    # Read static modality keys without constructing processors or downloading weights.
    tree = ast.parse((ROOT / "gr00t/experiment/data_config.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LiberoDataConfig")
    keys = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("video_keys", "state_keys", "action_keys")}
    assert keys["video_keys"] == ["video.front_view", "video.left_wrist_view"]
    assert keys["state_keys"] == ["state.eef_pos_absolute", "state.eef_rot_absolute", "state.gripper_close"]
    assert keys["action_keys"] == ["action.eef_pos_delta", "action.eef_rot_delta", "action.gripper_close"]
