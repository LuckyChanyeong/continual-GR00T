#!/bin/bash
#SBATCH -J libero-eval
#SBATCH -p gh
#SBATCH -N 1                  # 1 node per array element
#SBATCH --ntasks=1            # 1 task  (rank) on that node
#SBATCH --cpus-per-task=8     # CPU cores per task
#SBATCH -t 4:00:00
#SBATCH --array=0-3           # 9 jobs; use 0-8%3 to cap concurrency at 3
#SBATCH -o logs/%x_%A_%a.out  # stdout
#SBATCH -e logs/%x_%A_%a.err  # stderr

# Load Configuration
source $(dirname "$0")/config.sh

TASK_SUITE_NAME=$1
CL_TYPE=$2
CKPT_STEP=$3
N_MEMORIES=${4:-1000}
BATCH_SIZE=${5:-16}
ADD_NAME="${6:-}"

echo "TASK_SUITE_NAME: $TASK_SUITE_NAME"
echo "CL_TYPE: $CL_TYPE"
echo "CKPT_STEP: $CKPT_STEP"

export PYTHONPATH="$(pwd):${PYTHONPATH}"

mem_suffix=""
if [[ "$CL_TYPE" == "ER" ]]; then
  mem_suffix="_nmemory_${N_MEMORIES}"
fi


CKPT_DIR="${OUTPUT_PATH}/libero_cl/$TASK_SUITE_NAME/$CL_TYPE/step${CKPT_STEP}${mem_suffix}_bs${BATCH_SIZE}${ADD_NAME}/task_${SLURM_ARRAY_TASK_ID}/checkpoint-${CKPT_STEP}"

CKPT_PATH="$CKPT_DIR"

echo "[i] Evaluating: CKPT_STEP=$CKPT_STEP..."
echo "[i] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "[i] CKPT_PATH=$CKPT_PATH"

MAIN_PIDS=()

sleep 10

set -x

for task_id in $(seq 0 $((SLURM_ARRAY_TASK_ID))); do
	echo "Evaluating task $task_id"
	OUTPUT_DIR="$CKPT_DIR/eval_outputs"
	mkdir -p "$OUTPUT_DIR"
	
	python scripts/eval_policy_libero.py \
        --args.model-path $CKPT_PATH \
        --args.embodiment-tag "libero" \
        --args.data-config "libero" \
        --args.task-suite-name $TASK_SUITE_NAME \
        --args.video-out-path $OUTPUT_DIR \
        --args.task_idx=$task_id \
        >& "$OUTPUT_DIR/eval-${TASK_SUITE_NAME}-${task_id}.log" 
done
wait

set +x

echo "[i] Finished CKPT_STEP=$CKPT_STEP on GPU $SLURM_ARRAY_TASK_ID."
