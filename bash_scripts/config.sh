#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero"
export LIBERO_RAW_DATASET_PATH="${PROJECT_ROOT}/datasets/libero_raw"
export LIBERO_DATASET_PATH="${PROJECT_ROOT}/datasets/libero"
export OUTPUT_PATH="${PROJECT_ROOT}/outputs"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
