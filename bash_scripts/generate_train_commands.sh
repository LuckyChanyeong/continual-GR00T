#!/bin/bash

# Load Configuration
source $(dirname "$0")/config.sh

# Arguments
TASK_SUITE_NAME=${1}
STEPS=${2}
CL_TYPE=${3:-"ER"} # Default to ER if not provided
N_MEMORIES=${4:-1000} # Kept for compatibility, might not be used if EWC doesn't use replay buffer in same way, but passed to script
BATCH_SIZE=16
NUM_GPUS=1
SAVE_STEPS=${5:-$STEPS}
SEED=${6:-42}
TUNE_LLM=${7:-"False"}
TUNE_VISION=${8:-"False"}
ADD_NAME=${9:-"s${SEED}"}

# Define Task Suites
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
elif [ "$TASK_SUITE_NAME" == "libero_90" ]; then
    TASK_NAMES=(
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet"
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it"
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet"
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it"
        "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it"
        "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it"
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet"
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet"
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it"
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate"
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet"
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet"
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate"
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate"
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate"
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet"
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle"
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl"
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove"
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove"
        "KITCHEN_SCENE3_turn_on_the_stove"
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it"
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet"
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer"
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet"
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet"
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet"
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack"
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet"
        "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet"
        "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate"
        "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet"
        "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet"
        "KITCHEN_SCENE6_close_the_microwave"
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug"
        "KITCHEN_SCENE7_open_the_microwave"
        "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate"
        "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate"
        "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove"
        "KITCHEN_SCENE8_turn_off_the_stove"
        "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf"
        "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet"
        "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf"
        "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet"
        "KITCHEN_SCENE9_turn_on_the_stove"
        "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it"
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket"
        "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray"
        "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray"
        "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray"
        "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray"
        "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray"
        "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray"
        "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray"
        "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray"
        "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray"
        "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray"
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate"
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate"
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate"
        "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate"
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate"
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate"
        "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate"
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate"
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy"
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy"
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
        "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy"
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy"
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy"
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy"
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy"
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
        "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy"
        "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy"
        "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf"
        "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf"
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf"
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf"
    )
else
    echo "Invalid task suite name: $TASK_SUITE_NAME"
    exit 1
fi

# Build dataset paths string
DATASET_PATHS=""
for task in "${TASK_NAMES[@]}"; do
    DATASET_PATHS+="${LIBERO_DATASET_PATH}/${task} "
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

# Updated Output Dir
OUTPUT_DIR="${OUTPUT_PATH}/libero_cl/${TASK_SUITE_NAME}/${CL_TYPE}/step${STEPS}${mem_suffix}_bs$(($BATCH_SIZE * $NUM_GPUS))${ADD_NAME}"

# Construct the Command
COMMAND="WANDB_PROJECT=gr00t-training python scripts/gr00t_sequential_finetune.py \
    --dataset-path ${DATASET_PATHS} \
    --num-gpus ${NUM_GPUS} \
    --output-dir ${OUTPUT_DIR} \
    --max-steps ${STEPS} \
    --data-config libero \
    --embodiment-tag libero \
    --batch-size ${BATCH_SIZE} \
    --cl-type ${CL_TYPE} \
    --save-steps ${SAVE_STEPS} \
    --n-memories ${N_MEMORIES} \
    --seed ${SEED} \
    --run-name GR00T-N1.5-libero-cl_${TASK_SUITE_NAME}-${CL_TYPE}${ADD_NAME}-step${STEPS}-bs$(($BATCH_SIZE * $NUM_GPUS))${mem_suffix} \
    ${TUNE_LLM_FLAG} ${TUNE_VISION_FLAG}"

# Print the command
echo "$COMMAND"
