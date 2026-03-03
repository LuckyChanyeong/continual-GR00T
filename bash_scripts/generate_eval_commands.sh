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

BATCH_SIZE=16
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

# Experiment Directory
EXP_DIR="${OUTPUT_PATH}/libero_cl/${TASK_SUITE_NAME}/${CL_TYPE}/step${STEPS}${mem_suffix}_bs$(($BATCH_SIZE * $NUM_GPUS))${ADD_NAME}"

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
    
    CMD="mkdir -p ${OUTPUT_DIR} && python scripts/eval_policy_libero.py \
        --args.model-path ${CKPT_PATH} \
        --args.embodiment-tag libero \
        --args.data-config libero90 \
        --args.task-suite-name ${TASK_SUITE_NAME} \
        --args.video-out-path ${OUTPUT_DIR} \
        --args.task_idx=${task_id} \
        >& ${OUTPUT_DIR}/eval-${TASK_SUITE_NAME}-${task_id}.log"
    
    echo "$CMD"
    echo "" # Newline separator
done
