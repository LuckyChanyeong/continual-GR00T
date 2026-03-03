import collections
import dataclasses
import logging
import math
import os
import pathlib

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from pathlib import Path

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset, LeRobotSubsetDataset

import re

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    model_path: str = "nvidia/GR00T-N1.5-3B" # default model path
    embodiment_tag: str = "new_embodiment" # default embodiment tag
    data_config: str = "libero" # default data config
    denoising_steps: int = 4
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    task_idx: int = -1  # Task index to run

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, idx: {args.task_idx}")
    logging.info(f"Number of tasks in suite: {num_tasks_in_suite}")
    logging.info(f"Number of trials per task: {args.num_trials_per_task}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if (
        args.task_suite_name == "libero_spatial"
        or args.task_suite_name == "libero_spatial_new"
        or args.task_suite_name == "libero_spatial_ood"
    ):
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object" or args.task_suite_name == "libero_object_ood":
        max_steps = 280  # longest training demo has 254 steps
    elif (
        args.task_suite_name == "libero_goal"
        or args.task_suite_name == "libero_goal_new"
        or args.task_suite_name == "libero_goal_ood"
    ):
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")


    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()


    #### Task Order #####
    # Default order = sequential [0..num_tasks_in_suite-1]
    task_order = list(range(num_tasks_in_suite))

    match = re.search(r"order(\d+)", args.model_path)
    if match:
        order_str = match.group(1)  # e.g. "243"
        
        if order_str == "243":
            task_order = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]
        elif order_str == "324":
            task_order = [0, 1, 3, 2, 4, 5, 6, 7, 8, 9]
        elif order_str == "342":
            task_order = [0, 1, 3, 4, 2, 5, 6, 7, 8, 9]   
        elif order_str == "423":
            task_order = [0, 1, 4, 2, 3, 5, 6, 7, 8, 9] 
        
    base_eval_dir = Path(args.model_path).parent.parent / f"task_{args.task_idx}"
    print(f"Base evaluation directory: {base_eval_dir}")
    
    from gr00t import config
    base_dir_dataset = config.LIBERO_DATASET_PATH

    libero_task_map = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
     "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "put_both_moka_pots_on_the_stove",
        "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "turn_on_the_stove_and_put_the_moka_pot_on_it",
    ],
    }

    actual_task_idx = task_order[args.task_idx] if args.task_idx != -1 else -1
    train_dataset = LeRobotSingleDataset(
        dataset_path=f"{base_dir_dataset}/{libero_task_map[args.task_suite_name][actual_task_idx]}",
        modality_configs=modality_config,
        transforms=modality_transform,
        embodiment_tag=args.embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
        video_backend="cv2",
    )

    print(f"task_order: {task_order}")
    print(f"actual_task_idx: {actual_task_idx}")
    print(f"norm_stat dataset path: {train_dataset.dataset_path}")

    client = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        metadata_path=base_eval_dir,
        metadata_content=train_dataset.metadata,
    )
    logging.info(f"loaded model from {args.model_path}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if args.task_idx != -1 and task_id != args.task_idx:
            continue
        # Get task
        
        actual_task_idx = task_order[task_id]
        task = task_suite.get_task(actual_task_idx)
        
        print(f"Evaluating task {task_id}/{num_tasks_in_suite - 1}: {task.language}")
        print(f"Actual task index in training order: {actual_task_idx}")

        #### Check if we have enough videos already ####
        task_segment = task.language.replace(" ", "_")
        existing_videos = list(pathlib.Path(args.video_out_path).glob(f"rollout_{task_segment}_*.mp4"))
        if len(existing_videos) > 40:  # i.e., 50 or more
            logging.info(
                f"Detected {len(existing_videos)} videos for task '{task_segment}'. "
                "Skipping evaluation for this task."
            )
            continue

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(actual_task_idx)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            task_segment = task_description.replace(" ", "_")
            if os.path.exists(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_failure.mp4"):
                logging.info(f"Video already exists, skipping episode {episode_idx}...")
                total_episodes += 1
                task_episodes += 1
                continue
            elif os.path.exists(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_success.mp4"
            ):
                logging.info(f"Video already exists, skipping episode {episode_idx}...")
                total_episodes += 1
                task_episodes += 1
                total_successes += 1
                task_successes += 1
                continue

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # No need to resize for GR00T
                    # img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    # )
                    # wrist_img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    # )

                    # """  # Save images for debugging
                    if t == args.num_steps_wait and episode_idx == 0:
                        task_segment = task_description.replace(" ", "_")
                        image_prefix = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:02d}"
                        image_prefix.parent.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(f"{image_prefix}_img.png", img)
                        imageio.imwrite(f"{image_prefix}_wrist.png", wrist_img)
                    # """

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict

                        element = {
                            "video.front_view": np.array([img]),
                            "video.left_wrist_view": np.array([wrist_img]),
                            "state.eef_pos_absolute": obs["robot0_eef_pos"].reshape(1, -1),  # GR00T requries [horizon, state_dim]
                            "state.eef_rot_absolute": _quat2axisangle(obs["robot0_eef_quat"]).reshape(1, -1),
                            "state.gripper_close": obs["robot0_gripper_qpos"].reshape(1, -1),
                            "annotation.human.action.task_description": [str(task_description)],
                        }
                        # Query model to get action
                        action_chunk = client.get_action(element)
                        action_chunk = np.concatenate(
                            [
                                action_chunk["action.eef_pos_delta"],
                                action_chunk["action.eef_rot_delta"],
                                action_chunk["action.gripper_close"].reshape(-1, 1),
                            ],
                            axis=-1,
                        )
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    assert False
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=30,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    try:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
        logging.info(f"Total episodes: {total_episodes}")
    except ZeroDivisionError:
        logging.info("No episodes completed in this run.")
    result_save_path = pathlib.Path(args.video_out_path) / f"{args.task_idx}_results.txt"
    
    if match:
        result_save_path = pathlib.Path(args.video_out_path) / f"trained_id{args.task_idx}_actual_id{actual_task_idx}_results.txt"
    
    try:    
        with open(result_save_path, "w") as f:
            f.write(f"Total success rate: {float(total_successes) / float(total_episodes)}\n")
            f.write(f"Total episodes: {total_episodes}")
        logging.info(f"Results saved to {result_save_path}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
