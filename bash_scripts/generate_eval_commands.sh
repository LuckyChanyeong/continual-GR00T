#!/bin/bash

# Load Configuration
source $(dirname "$0")/config.sh

# Arguments (Matching generate_ewc_commands.sh for path reconstruction)
TASK_SUITE_NAME=${1}
STEPS=${2}
CL_TYPE=${3:-"ER"}
N_MEMORIES=${4:-1000}
SAVE_STEPS=${5:-$STEPS} # Not used for path, but keeps arg alignment
SEED=${6:-42}
TUNE_LLM=${7:-"False"}
TUNE_VISION=${8:-"False"}
ADD_NAME=${9:-"s${SEED}"}

# New Args for Eval
CURRENT_TASK_ID=${10:-0} # The task id identifying the checkpoint to evaluate
CKPT_STEP=${11:-$STEPS} # The specific step checkpoint to load
BATCH_SIZE=${12:-16}
GRAD_ACCUMULATION_STEPS=${13:-1}
OPTIMIZER=${14:-"adamw_torch"}
NUM_TRIALS_PER_TASK=${15:-50}
ENABLE_FOXGLOVE=${16:-"False"}
FOXGLOVE_HOST=${17:-"127.0.0.1"}
FOXGLOVE_PORT=${18:-8765}
SAVE_VIDEO=${19:-"True"}

NUM_GPUS=1

mem_suffix=""
if [[ "$CL_TYPE" == "ER" ]]; then
  mem_suffix="_nmemory_${N_MEMORIES}"
fi

tune_suffix=""
if [[ "$TUNE_LLM" == "True" ]]; then
  tune_suffix="_tune_llm"
fi
if [[ "$TUNE_VISION" == "True" ]]; then
  tune_suffix="${tune_suffix}_tune_vision"
fi

ADD_NAME="${ADD_NAME}${tune_suffix}"

foxglove_args=""
if [[ "$ENABLE_FOXGLOVE" == "True" ]]; then
  foxglove_args="--args.foxglove --args.foxglove-host ${FOXGLOVE_HOST} --args.foxglove-port ${FOXGLOVE_PORT}"
fi

video_args=""
if [[ "$SAVE_VIDEO" != "True" ]]; then
  video_args="--args.no-save-video"
fi

batch_suffix="bs$(($BATCH_SIZE * $NUM_GPUS))"
if [[ "$GRAD_ACCUMULATION_STEPS" -ne 1 ]]; then
  batch_suffix="${batch_suffix}_ga${GRAD_ACCUMULATION_STEPS}"
fi
if [[ "$OPTIMIZER" != "adamw_torch" ]]; then
  batch_suffix="${batch_suffix}_${OPTIMIZER}"
fi

# Experiment Directory
EXP_DIR="${OUTPUT_PATH}/libero_cl/${TASK_SUITE_NAME}/${CL_TYPE}/step${STEPS}${mem_suffix}_${batch_suffix}${ADD_NAME}"

# Checkpoint Path
CKPT_PATH="${EXP_DIR}/task_${CURRENT_TASK_ID}/checkpoint-${CKPT_STEP}"

# Base Output Directory for Eval Logs/Videos
EVAL_OUTPUT_BASE="${OUTPUT_PATH}/eval_output/libero_cl/${TASK_SUITE_NAME}/${CL_TYPE}/step${STEPS}_task${CURRENT_TASK_ID}_ckpt${CKPT_STEP}"


# --- Generate Commands ---

echo "# Evaluation Commands for Task Suite: $TASK_SUITE_NAME, Training Task: $CURRENT_TASK_ID, Checkpoint: $CKPT_STEP"
echo "# Checkpoint Path: $CKPT_PATH"
echo ""

for task_id in $(seq 0 ${CURRENT_TASK_ID}); do
    OUTPUT_DIR="${EVAL_OUTPUT_BASE}/task_${task_id}"

    # Embed the runtime environment because eval "$(bash ...)" cannot inherit
    # exports made by this generator's child shell.
    CMD="mkdir -p ${OUTPUT_DIR} && env \
        PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/LIBERO:${PYTHONPATH:-} \
        LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH} \
        LIBERO_RAW_DATASET_PATH=${LIBERO_RAW_DATASET_PATH} \
        LIBERO_DATASET_PATH=${LIBERO_DATASET_PATH} \
        MUJOCO_GL=${MUJOCO_GL} \
        ${PROJECT_ROOT}/.venv/bin/python scripts/eval_policy_libero.py \
        --args.model-path ${CKPT_PATH} \
        --args.embodiment-tag libero \
        --args.data-config libero \
        --args.task-suite-name ${TASK_SUITE_NAME} \
        --args.num-trials-per-task ${NUM_TRIALS_PER_TASK} \
        ${foxglove_args} \
        ${video_args} \
        --args.video-out-path ${OUTPUT_DIR} \
        --args.task_idx=${task_id} \
        > >(tee ${OUTPUT_DIR}/eval-${TASK_SUITE_NAME}-${task_id}.log) 2>&1"
    
    echo "$CMD"
    echo "" # Newline separator
done
