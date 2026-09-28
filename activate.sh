#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${PROJECT_ROOT}/.venv/bin/activate"
source "${PROJECT_ROOT}/bash_scripts/config.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"

cd "${PROJECT_ROOT}"
