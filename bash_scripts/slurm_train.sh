#!/bin/bash
#SBATCH -J gr00t_libero_cl 
#SBATCH -p gh
#SBATCH -N 1                  # 1 node per array element
#SBATCH --ntasks=1            # 1 task  (rank) on that node
#SBATCH --cpus-per-task=8     # CPU cores per task
#SBATCH -t 40:00:00
#SBATCH -o logs/%x_%A.out  # stdout
#SBATCH -e logs/%x_%A.err  # stderr

# Load Configuration
source $(dirname "$0")/config.sh

TASK_SUITE_NAME=${1}
STEPS=${2}
CL_TYPE=${3}
N_MEMORIES=${4:-1000}
BATCH_SIZE=16
NUM_GPUS=1
SAVE_STEPS=${5:-$STEPS}
SEED=${6:-42} # Use a random seed if not provided
TUNE_LLM=${7:-"False"}
TUNE_VISION=${8:-"False"}
ADD_NAME=${9:-"s${SEED}"}

if [ "$TASK_SUITE_NAME" == "libero_10" ]; then
    TASK_NAMES=(
        "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
        "put_both_moka_pots_on_the_stove"
        "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
        "put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
        "put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
        "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
        "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
        "put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
        "turn_on_the_stove_and_put_the_moka_pot_on_it"
    )
elif [ "$TASK_SUITE_NAME" == "libero_object" ]; then
    TASK_NAMES=(
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
        "pick_up_the_butter_and_place_it_in_the_basket"
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
        "pick_up_the_cream_cheese_and_place_it_in_the_basket"
        "pick_up_the_ketchup_and_place_it_in_the_basket"
        "pick_up_the_milk_and_place_it_in_the_basket"
        "pick_up_the_orange_juice_and_place_it_in_the_basket"
        "pick_up_the_salad_dressing_and_place_it_in_the_basket"
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket"
    )
elif [ "$TASK_SUITE_NAME" == "libero_spatial" ]; then
    TASK_NAMES=(
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
    )
elif [ "$TASK_SUITE_NAME" == "libero_goal" ]; then
    TASK_NAMES=(
        "open_the_middle_drawer_of_the_cabinet"
        "put_the_bowl_on_the_stove"
        "put_the_wine_bottle_on_top_of_the_cabinet"
        "open_the_top_drawer_and_put_the_bowl_inside"
        "put_the_bowl_on_top_of_the_cabinet"
        "push_the_plate_to_the_front_of_the_stove"
        "put_the_cream_cheese_in_the_bowl"
        "turn_on_the_stove"
        "put_the_bowl_on_the_plate"
        "put_the_wine_bottle_on_the_rack"
    )
else
    echo "Invalid task suite name"
    exit 1
fi

# BASE_PATH: change it according to the path of the dataset
BASE_PATH=${LIBERO_DATASET_PATH:-./datasets/libero}

# Build dataset paths string
DATASET_PATHS=""
for i in {0..9}; do
    DATASET_PATHS+="${BASE_PATH}/${TASK_NAMES[$i]} "
done

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

if [ "$TUNE_LLM" = true ]; then
    TUNE_LLM_FLAG="--tune-llm"
else
    TUNE_LLM_FLAG=""
fi
if [ "$TUNE_VISION" = true ]; then
    TUNE_VISION_FLAG="--tune-visual"
else
    TUNE_VISION_FLAG=""
fi

ADD_NAME="${ADD_NAME}${tune_suffix}"

set -x
WANDB_PROJECT=gr00t-training python scripts/gr00t_sequential_finetune.py \
    --dataset-path ${DATASET_PATHS} \
    --num-gpus ${NUM_GPUS} \
    --output-dir ${OUTPUT_PATH:-./output} \
    --max-steps ${STEPS} \
    --data-config libero \
    --embodiment-tag libero \
    --batch-size ${BATCH_SIZE} \
    --cl-type ${CL_TYPE} \
    --save-steps ${SAVE_STEPS} \
    --n-memories ${N_MEMORIES} \
    --seed ${SEED} \
    --run-name GR00T-N1.5-libero-cl_${TASK_SUITE_NAME}-${CL_TYPE}${ADD_NAME}-step${STEPS}-bs$(($BATCH_SIZE * $NUM_GPUS))${mem_suffix} \
    ${TUNE_LLM_FLAG} ${TUNE_VISION_FLAG} 
set +x
