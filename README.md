# Continual Learning Bash Scripts

This directory contains utility scripts for generating training and evaluation commands, as well as Slurm submission scripts for cluster environments.

See `README_GR00T.md` in the root directory for more information on GR00T installation and setup.

## Configuration

Before running any scripts, ensure that your paths are correctly configured in `bash_scripts/config.sh`. This file is sourced by all other scripts to ensure consistent dataset and output locations.

```bash
# bash_scripts/config.sh
export LIBERO_DATASET_PATH="/path/to/libero/dataset"
export OUTPUT_PATH="/path/to/experiments/output"
```

---

## Command Generation Scripts

These scripts are used to generate the full Python commands required to run experiments. You can pipe the output to a file or execute it directly.

### 1. Training Command Generation
`generate_train_commands.sh`

**Usage:**
```bash
./bash_scripts/generate_train_commands.sh <TASK_SUITE_NAME> <STEPS> [CL_TYPE] [N_MEMORIES] [SAVE_STEPS] [SEED] [TUNE_LLM] [TUNE_VISION] [ADD_NAME]
```

- **Example:** `./bash_scripts/generate_train_commands.sh libero_10 10000 ER 1000 10000 42 False False s42`
- **Arguments:**
    - `TASK_SUITE_NAME`: `libero_10`, `libero_object`, `libero_spatial`, `libero_goal`, `libero_90`
    - `STEPS`: Total training steps.
    - `CL_TYPE`: (Optional) Continual Learning type (default: `ER`).
    - `N_MEMORIES`: (Optional) Memory buffer size for ER (default: `1000`).

### 2. Evaluation Command Generation
`generate_eval_commands.sh`

**Usage:**
```bash
./bash_scripts/generate_eval_commands.sh <TASK_SUITE_NAME> <STEPS> [CL_TYPE] [N_MEMORIES] [SAVE_STEPS] [SEED] [TUNE_LLM] [TUNE_VISION] [ADD_NAME] [CURRENT_TASK_ID] [CKPT_STEP]
```

- **Example:** `./bash_scripts/generate_eval_commands.sh libero_10 10000 ER 1000 10000 42 False False s42 6 10000`
- Generates evaluation commands for all tasks up to `CURRENT_TASK_ID` using the checkpoint at `CKPT_STEP`.
- For the above command, it will evaluate task `6` checkpoint using step `10000`, on all tasks from `0` to `6`.

---

## Slurm Submission Scripts

For running on a cluster using the Slurm workload manager.

### 1. Slurm Training
`slurm_train.sh`

**Usage:**
```bash
sbatch bash_scripts/slurm_train.sh <TASK_SUITE_NAME> <STEPS> <CL_TYPE> [N_MEMORIES] [SAVE_STEPS] [SEED] [TUNE_LLM] [TUNE_VISION] [ADD_NAME]
```

### 2. Slurm Evaluation
`slurm_eval.sh`

Uses Slurm job arrays to parallelize evaluation across tasks.

**Usage:**
```bash
sbatch bash_scripts/slurm_eval.sh <TASK_SUITE_NAME> <CL_TYPE> <CKPT_STEP> [N_MEMORIES] [BATCH_SIZE] [ADD_NAME]
```

---

## Note on Python Environment
When running these scripts, ensure your `PYTHONPATH` includes the project root:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
The scripts automatically attempt to set this when pulling configuration from `gr00t/config.py`.
