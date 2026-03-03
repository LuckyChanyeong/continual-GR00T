# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import json
import os
import warnings
from collections import defaultdict
from glob import glob
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robocasa
import robosuite
from gymnasium.wrappers import TimeLimit
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy

import robocasa.utils.robomimic.robomimic_dataset_utils as FileUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils

import collections

from robosuite.wrappers import GymWrapper
from gr00t.eval.wrappers.data_collection_wrapper import DataCollectionWrapper

warnings.simplefilter("ignore", category=FutureWarning)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        actions_abs = []
        rewards = []
        # success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            rewards.extend(dic["rewards"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])
            # success = success or dic["successful"]

        if len(states) == 0:
            continue

        # # Add only the successful demonstration to dataset
        # if success:

        # print("Demonstration is successful and has been saved")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as f:
                ep_meta = f.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("rewards", data=np.array(rewards))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))

        # else:
        #     pass
        #     # print("Demonstration is unsuccessful and has NOT been saved")

    print("{} successful demos so far".format(num_eps))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["robocasa_version"] = robocasa.__version__
    grp.attrs["robosuite_version"] = robosuite.__version__
    grp.attrs["mujoco_version"] = mujoco.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

    return hdf5_path


"""
Example command:

python scripts/eval_policy_robocasa.py --host localhost --port 5555
    --action_horizon 16
    --video_backend decord
    --dataset_path demo_data/robot_sim.PickNPlace/
    --embodiment_tag gr1
    --data_config gr1_arms_waist
    --env_name CloseDrawer
    --num_episodes 10
provide --model_path to load up the model checkpoint in this script.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="host")
    parser.add_argument("--port", type=int, default=5555, help="port")
    parser.add_argument(
        "--data_config",
        type=str,
        default="gr1_arms_only",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--video_backend", type=str, default="decord")
    parser.add_argument("--dataset_path", type=str, default="demo_data/robot_sim.PickNPlace/")
    
    parser.add_argument("--dataset_path_hdf5", type=str)
    
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    ## When using a model instead of client-server mode.
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps if model_path is provided",
        default=4,
    )

    # robocasa env and evaluation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the robocasa environment",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CloseDrawer",
        help="Name of the robocasa environment to load",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=300,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the video",
    )

    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )
    # parser.add_argument(
    #     "--data_collection_path",
    #     type=str,
    #     default=None,
    #     help="Path to save the data collection",
    # )
    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument(
        "--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    )
    parser.add_argument("--generative_textures", action="store_true", help="Use generative textures")

    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    all_gt_actions = []
    all_pred_actions = []

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print(modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)

    # load robocasa env
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    env_name = args.env_name
    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
        "generative_textures": "100p",
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment
    if env_name in ["Lift"]:  # add other non-kitchen tasks here
        if args.obj_groups is not None:
            print(
                "Specifying 'obj_groups' in non-kitchen environment does not have an effect."
            )
    else:
        config["layout_ids"] = args.layout
        config["style_ids"] = args.style
        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})

        config["translucent_robot"] = True

        # by default use obj instance split A
        config["obj_instance_split"] = "A"
        # config["obj_instance_split"] = None
        # config["obj_registries"] = ("aigen",)

    # Grab reference to controller config and convert it to json-encoded string
    # env_info = json.dumps(config)


    # New way of creating envs
    
    DEFAULT_EVAL_UPDATE_KWARGS =  {
        "generative_textures": None,
        "randomize_cameras": False,
        "obj_instance_split": "B",
        "layout_ids": None,
        "style_ids": None,
        "scene_split": None,
        "layout_and_style_ids": [
                        [
                            1,
                            1
                        ],
                        [
                            2,
                            2
                        ],
                        [
                            4,
                            4
                        ],
                        [
                            6,
                            9
                        ],
                        [
                            7,
                            10
                        ]
                    ],
        "camera_heights": 128,
        "camera_widths": 128,
    }
    
    SHAPE_META = {
        "obs": {
            "robot0_agentview_left_image": {
                "shape": [128, 128, 3],
                "type": "rgb"
            },
            "robot0_eye_in_hand_image": {
                "shape": [128, 128, 3],
                "type": "rgb"
            },
            "robot0_agentview_right_image": {
                "shape": [128, 128, 3],
                "type": "rgb"
            },
            "robot0_base_to_eef_pos": {
                "shape": [3]
                # type default: low_dim (not explicitly listed)
            },
            "robot0_base_to_eef_quat": {
                "shape": [4]
            },
            "robot0_gripper_qpos": {
                "shape": [2]
            },
        },
        "action": {
            "shape": [12]
        }
    }
    
    def _get_robocasa_env(env_meta, shape_meta, enable_render=True):
        """Initializes and returns the LIBERO environment, along with the task description."""
        modality_mapping = collections.defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        env_name = env_meta['env_name']
        env_name = env_name[3:] if env_name.startswith('MG_') else env_name
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_name,
            render=False, 
            render_offscreen=enable_render,
            use_image_obs=enable_render, 
        )
        
        ### Changes for gr00t
        env = env.env
        env = DataCollectionWrapper(env, directory="tmp", collect_freq=1, flush_freq=100)
        env = GymWrapper(
            env,
            flatten_obs=False,
            keys=[
                "robot0_base_pos",
                "robot0_base_quat",
                "robot0_eef_pos",
                "robot0_base_to_eef_pos",
                "robot0_eef_quat",
                "robot0_base_to_eef_quat",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel",
                "robot0_joint_pos",
                "robot0_joint_pos_cos",
                "robot0_joint_pos_sin",
                "robot0_joint_vel",
                "robot0_agentview_left_image",
                "robot0_agentview_right_image",
                "robot0_eye_in_hand_image",
            ],
        )    
        
        return env
    
    env_meta = FileUtils.get_env_metadata_from_dataset(
            args.dataset_path_hdf5)
    env_meta['env_kwargs']['use_object_obs'] = False
    env_meta['env_kwargs'].update(DEFAULT_EVAL_UPDATE_KWARGS)
    env = _get_robocasa_env(env_meta, SHAPE_META, enable_render=True)
    
    # env = load_robocasa_gym_env(
    #     args.env_name,
    #     seed=args.seed,
    #     # directory=Path(args.data_collection_path),
    #     generative_textures="100p" if args.generative_textures else None,
    # )
    env = RoboCasaWrapper(env)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    record_video = args.video_path is not None
    if record_video:
        video_base_path = Path(args.video_path)
        # video_base_path.mkdir(parents=True, exist_ok=True)
        episode_trigger = lambda t: t % 1 == 0  # noqa
        env = RecordVideo(env, video_base_path, disable_logger=True, episode_trigger=episode_trigger, fps=20)

    env = MultiStepWrapper(
        env,
        video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1),
        n_action_steps=args.action_horizon,
    )

    # postprocess function of action, to handle the case where number of dimensions are not the same
    def postprocess_action(action):
        new_action = {}
        for k, v in action.items():
            if v.ndim == 1:
                new_action[k] = v[..., None]
            else:
                new_action[k] = v
        return new_action

    # main evaluation loop
    stats = defaultdict(list)
    for i in trange(args.num_episodes):
        pbar = tqdm(
            total=args.max_episode_steps, desc=f"Episode {i + 1} / {env.unwrapped.get_ep_meta()['lang']}", leave=False
        )
        obs, info = env.reset()
        done = False
        step = 0
        while not done:
            action = policy.get_action(obs)
            post_action = postprocess_action(action)
            next_obs, reward, terminated, truncated, info = env.step(post_action)
            done = terminated or truncated
            step += args.action_horizon
            obs = next_obs
            pbar.update(args.action_horizon)
        add_to(stats, flatten({"is_success": info["is_success"]}))
        pbar.close()

    env.close()

    # print("Change collected data to hdf5 format")
    # hdf5_path = gather_demonstrations_as_hdf5(
    #     args.data_collection_path, args.data_collection_path, env_info
    # )
    # convert_to_robomimic_format(hdf5_path)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    print(stats)

    exit()
