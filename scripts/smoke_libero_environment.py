"""Small GPU-0/simulator smoke test; no policy weights or demonstration datasets."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".config/libero"))


def main():
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from gr00t.experiment.continual import validate_task_catalog

    validate_task_catalog(benchmark)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Assigned GPU 0 unavailable; refusing fallback")
    print("GPU 0:", torch.cuda.get_device_name(0))
    a = torch.ones(4, device="cuda:0")
    assert (a + a).sum().item() == 8
    suite = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=64, camera_widths=64,
                             render_gpu_device_id=0)
    try:
        env.seed(7)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        assert obs["agentview_image"].shape == (64, 64, 3)
        obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
        print("EGL render + initial state + one dummy step OK:", obs["agentview_image"].shape)
    finally:
        env.close()


if __name__ == "__main__":
    main()
