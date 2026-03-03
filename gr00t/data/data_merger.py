# lerobot_dataset_manager.py
# Copyright 2024-2025 The HuggingFace Inc. team and contributors.
# Licensed under the Apache-2.0 license.

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd  # type: ignore
from tqdm import tqdm

# --- Constants ---
PAD = 6  # Padding for episode numbers (e.g., 000032)
CHUNK_NAME_DEFAULT = "chunk-000"  # Default chunk name, primarily for CLI convenience
MERGE_NUM_KEYS = ["total_tasks"]  # For merge_info
DELETE_STEM_RE = re.compile(r"^episode_(\d{6})$")
DELETE_PATCH_KEYS = {"episode_index", "index"}  # For delete _patch


_LIBERO_TASK_SUITES: Dict[str, List[str]] = {
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
    
    
    #### Robocasa ####
    "pnp": [
        "PnPSinkToCounter",
        "PnPCounterToMicrowave",
        "PnPMicrowaveToCounter",
        "PnPCounterToCab",
        "PnPStoveToCounter",
        "PnPCounterToSink",
        "PnPCabToCounter",
        "PnPCounterToStove",
    ],

    "door": [
        "CloseDoubleDoor",
        "CloseDrawer",
        "CloseSingleDoor",
        "OpenDoubleDoor",
        "OpenDrawer",
        "OpenSingleDoor",
    ],

    "all": [
        "CloseDoubleDoor",
        "CloseDrawer",
        "CloseSingleDoor",
        "CoffeePressButton",
        "CoffeeServeMug",
        "CoffeeSetupMug",
        "OpenDoubleDoor",
        "OpenDrawer",
        "OpenSingleDoor",
        "PnPCabToCounter",
        "PnPCounterToCab",
        "PnPCounterToMicrowave",
        "PnPCounterToSink",
        "PnPCounterToStove",
        "PnPMicrowaveToCounter",
        "PnPSinkToCounter",
        "PnPStoveToCounter",
        "TurnOffMicrowave",
        "TurnOffSinkFaucet",
        "TurnOffStove",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
        "TurnOnStove",
        "TurnSinkSpout",
    ],

    "all-randomized": [
        "PnPSinkToCounter",
        "CoffeeSetupMug",
        "TurnOffSinkFaucet",
        "TurnSinkSpout",
        "PnPCounterToStove",
        "PnPCounterToCab",
        "TurnOffMicrowave",
        "PnPCounterToMicrowave",
        "TurnOffStove",
        "PnPStoveToCounter",
        "CloseDrawer",
        "PnPCounterToSink",
        "TurnOnSinkFaucet",
        "OpenDoubleDoor",
        "PnPCabToCounter",
        "CloseSingleDoor",
        "TurnOnStove",
        "CoffeeServeMug",
        "PnPMicrowaveToCounter",
        "OpenDrawer",
        "OpenSingleDoor",
        "CloseDoubleDoor",
        "CoffeePressButton",
        "TurnOnMicrowave",
    ],
    
}


def build_multitask_dataset(
    task_suite_name: str,
    datasets_root: Path,
    output_dir: Path,
    *,
    chunk_name: str = CHUNK_NAME_DEFAULT,
    num_episodes: int = 1_000_000,  # “big number” → merge everything
    verbose: bool = False,
) -> None:
    """
    Assembles a multi-task dataset for *task_suite_name*.

    The function assumes:
      • Every *single-task* dataset already exists at
        ``datasets_root / slugify(task_name)``.
      • You want to include **all** episodes those datasets contain.

    Parameters
    ----------
    task_suite_name
        One of ``libero_10``, ``libero_object``, ``libero_spatial``,
        ``libero_goal``.
    datasets_root
        Directory that holds the individual task folders.
    output_dir
        Where the new multi-task dataset will be written.
    chunk_name, num_episodes, verbose
        Passed straight through to ``DatasetManager.merge_datasets``.
    """
    if task_suite_name not in _LIBERO_TASK_SUITES:  # Defensive guard
        raise ValueError(
            f"Unknown task-suite '{task_suite_name}'. "
            f"Expected one of: {', '.join(_LIBERO_TASK_SUITES)}"
        )

    task_names = _LIBERO_TASK_SUITES[task_suite_name]

    # Resolve the *single-task* dataset folders we are going to merge
    dataset_paths = [datasets_root / t for t in task_names]
    missing = [p for p in dataset_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The following task datasets are missing:\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    dm = DatasetManager()
    dm.merge_datasets(
        " ".join(str(p) for p in dataset_paths),  # space-separated list
        output_dir=output_dir,
        chunk_path=chunk_name,
        num_episodes=num_episodes,
        verbose=verbose,
    )


def slugify(txt: str) -> str:
    """
    Slugify a string to make it a valid filename.
    Example: "Hello World" -> "hello_world"
    """
    return re.sub(r"[^a-z0-9]+", "_", txt.lower()).strip("_")


class DatasetManager:
    """
    Manages Lerobot datasets, allowing operations like merging and deleting episodes.
    """

    # ─────────────────────────────────── Static Utilities ────────────────────────────────── #
    @staticmethod
    def _extract_idx_from_name(name_with_index: str) -> int:
        """Extracts a numerical index from a filename stem or full name."""
        num_re = re.compile(r"(\d+)(?=\.parquet$|\.mp4$|$)")
        match = num_re.search(name_with_index)
        if not match:
            raise ValueError(f"Impossible de trouver un index numérique dans {name_with_index}")
        return int(match.group(1))

    @staticmethod
    def _natural_sort_paths(paths: Iterable[Path]) -> List[Path]:
        """Sorts an iterable of Path objects naturally based on extracted numerical indices."""
        return sorted(paths, key=lambda p: DatasetManager._extract_idx_from_name(p.name))

    @staticmethod
    def safe_mkdir(path: Path) -> None:
        """Creates a directory if it doesn't exist, including parent directories."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_jsonl(path: Path) -> List[Dict]:
        """Reads a JSONL file and returns a list of dictionaries."""
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def write_jsonl(objs: Iterable[Dict], path: Path) -> None:
        """Writes an iterable of dictionaries to a JSONL file."""
        with path.open("w") as f:
            for o in objs:
                f.write(json.dumps(o, separators=(",", ":")) + "\n")

    # --- Utilities for MERGE ---
    @staticmethod
    def _shift_any_positive_recursive(obj: Any, offset: int) -> Any:
        """Recursively shifts integer values in a JSON-like object by a positive offset (for merging)."""
        if isinstance(obj, int):
            return obj + offset
        if isinstance(obj, list):
            return [DatasetManager._shift_any_positive_recursive(x, offset) for x in obj]
        if isinstance(obj, dict):
            return {k: DatasetManager._shift_any_positive_recursive(v, offset) for k, v in obj.items()}
        return obj

    # --- Utilities for DELETE ---
    @staticmethod
    def _ep_id_from_stem(name: str) -> Optional[int]:
        m = DELETE_STEM_RE.match(name)
        return int(m.group(1)) if m else None

    @staticmethod
    def _add_offset_for_delete(val: Any, off: int):
        if isinstance(val, int):
            return val + off
        if isinstance(val, list):
            return [DatasetManager._add_offset_for_delete(x, off) for x in val]
        return val

    @staticmethod
    def _patch_indices_recursive_negative(obj: Any, off: int):
        """Recursively shifts specific integer values in a JSON-like object by a negative offset (for deleting)."""
        if isinstance(obj, dict):
            return {
                k: (
                    DatasetManager._add_offset_for_delete(v, off)
                    if k in DELETE_PATCH_KEYS
                    else DatasetManager._patch_indices_recursive_negative(v, off)
                )
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [DatasetManager._patch_indices_recursive_negative(x, off) for x in obj]
        return obj

    # ─────────────────────────────────── MERGE Operation ─────────────────────────────────── #

    def merge_datasets(
        self, dataset_paths_str: str, output_dir: Path, chunk_path: str, num_episodes: int, verbose: bool = False
    ):
        """
        Merges multiple Lerobot datasets into a new output directory.
        """
        dataset_paths = [Path(p.strip()) for p in dataset_paths_str.strip().split() if p.strip()]
        # chunk_paths = [f"chunk-{i:03d}" for i in range(len(dataset_paths))]
        chunk_paths = [chunk_path] * len(dataset_paths) 
        
        if not dataset_paths:
            print("No dataset paths provided for merging.")
            return

        if verbose:
            print(f"Starting merge operation. Output directory: {output_dir}")

        meta_dst_dir = output_dir / "meta"

        data_dst_dirs = [output_dir / "data" / chunk_path for chunk_path in chunk_paths]
        # video_dst_chunk_roots = [output_dir / "videos" / chunk_path for chunk_path in chunk_paths]
        video_dst_chunk_roots = [output_dir / "videos" / chunk_path] * len(dataset_paths)
       
        self.safe_mkdir(meta_dst_dir)
        for data_dst_dir in data_dst_dirs:
            self.safe_mkdir(data_dst_dir)
        for video_dst_chunk_root in video_dst_chunk_roots:
            self.safe_mkdir(video_dst_chunk_root.parent)
            self.safe_mkdir(video_dst_chunk_root)

        cumulative_episode_offset_parquets = 0
        cumulative_frame_offset_parquets = 0
        total_parquets_processed_overall = 0
        actual_episode_counts_per_dataset = []
        actual_frame_counts_per_dataset = []

        if verbose:
            print("--- Processing Parquet Files and Determining Episode Counts ---")
        for i, (dataset_path, data_dst_dir) in enumerate(zip(dataset_paths, data_dst_dirs)):
            processed_eps, processed_frames = self._copy_parquet_and_update_indices_for_merge(
                dataset_path,
                data_dst_dir,
                chunk_path,
                cumulative_episode_offset_parquets,
                cumulative_frame_offset_parquets,
                num_episodes,
                verbose,
            )
            if verbose:
                print(f"  Processed {processed_eps} Parquet episode files from {dataset_path}.")
            total_parquets_processed_overall += processed_eps
            actual_episode_counts_per_dataset.append(processed_eps)
            actual_frame_counts_per_dataset.append(processed_frames)
            cumulative_episode_offset_parquets += processed_eps
            cumulative_frame_offset_parquets += processed_frames

        if verbose:
            print("\n--- Processing Metadata Files ---")
        self._merge_all_meta_files(
            dataset_paths, meta_dst_dir, actual_episode_counts_per_dataset, actual_frame_counts_per_dataset, verbose
        )

        if verbose:
            print("\n--- Processing Video Files ---")
        self._copy_all_videos_for_merge(
            dataset_paths, video_dst_chunk_roots, chunk_path, actual_episode_counts_per_dataset, verbose
        )

        final_info_path = meta_dst_dir / "info.json"
        final_total_episodes = "N/A"
        if final_info_path.exists():
            try:
                final_info = json.loads(final_info_path.read_text())
                final_total_episodes = final_info.get("total_episodes", "N/A")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not parse {final_info_path} to get final total episodes: {e}")

        print(
            "\n✅ Merge finished!\n"
            f"  • Total Parquet files processed: {total_parquets_processed_overall}\n"
            f"  • Total episodes in merged dataset (from info.json): {final_total_episodes}\n"
            f"  • Output directory: {output_dir}"
        )

    def _copy_parquet_and_update_indices_for_merge(
        self,
        src_root: Path,
        dst_data_dir: Path,
        chunk_name: str,
        episode_idx_offset: int,
        frame_idx_offset: int,
        num_episodes: int,
        verbose: bool,
    ) -> int:
        src_chunk_dir = src_root / "data" / chunk_name
        if not src_chunk_dir.exists():
            if verbose:
                print(f"Source chunk directory not found: {src_chunk_dir}")
            return 0
        src_files = self._natural_sort_paths(src_chunk_dir.glob("episode_*.parquet"))
        if not src_files:
            if verbose:
                print(f"No Parquet files found in {src_chunk_dir}")
            return 0

        count_processed = 0
        frames_processed = 0
        for src_file_path in src_files[:num_episodes]:
            original_episode_idx = self._extract_idx_from_name(src_file_path.name)
            new_episode_global_idx = original_episode_idx + episode_idx_offset
            dst_file_path = dst_data_dir / f"episode_{new_episode_global_idx:0{PAD}d}.parquet"
            try:
                df = pd.read_parquet(src_file_path)
                if "episode_index" in df.columns:
                    df["episode_index"] = new_episode_global_idx
                if "index" in df.columns:
                    df["index"] = df["index"] + frame_idx_offset
                if "frame_index" in df.columns:
                    df["frame_index"] = df["frame_index"] + frame_idx_offset
                df.to_parquet(dst_file_path)
                count_processed += 1
                frames_processed += len(df)
            except Exception as e:
                print(f"Error processing Parquet {src_file_path} to {dst_file_path}: {e}")
                if dst_file_path.exists():
                    dst_file_path.unlink(missing_ok=True)
        return count_processed, frames_processed

    def _merge_all_meta_files(
        self,
        dataset_paths: List[Path],
        meta_dst_dir: Path,
        actual_episode_counts: List[int],
        actual_frame_counts: List[int],
        verbose: bool,
    ):
        
        total_episodes_all = sum(actual_episode_counts)
        
        current_meta_episode_offset = 0
        # ep_stats_out = meta_dst_dir / "episodes_stats.jsonl"
        ep_out = meta_dst_dir / "episodes.jsonl"
        tasks_out = meta_dst_dir / "tasks.jsonl"
        info_out = meta_dst_dir / "info.json"
        modality_out = meta_dst_dir / "modality.json"

        for p in [ep_out, tasks_out, info_out]:
            p.unlink(missing_ok=True)

        for i, dataset_path in enumerate(dataset_paths):
            src_meta_dir = dataset_path / "meta"
            eps_in_this_ds_for_meta = actual_episode_counts[i]
            if not src_meta_dir.exists():
                if verbose:
                    print(f"  Warning: Meta dir {src_meta_dir} not found.")
                current_meta_episode_offset += eps_in_this_ds_for_meta
                continue
                
            # Merge episodes.jsonl
            src_ep = src_meta_dir / "episodes.jsonl"
            if src_ep.exists():
                base_data = self.read_jsonl(ep_out) if ep_out.exists() else []
                new_data = self.read_jsonl(src_ep)[: actual_episode_counts[i]]
                for r_new in new_data:  # Similar shifting as episodes_stats
                    r_new["episode_index"] += current_meta_episode_offset
                    if "index" in r_new:
                        idx_val = r_new["index"]
                        r_new["index"] = (
                            [x + current_meta_episode_offset for x in idx_val]
                            if isinstance(idx_val, list)
                            else idx_val + current_meta_episode_offset
                        )
                self.write_jsonl(base_data + new_data, ep_out)

            # Merge tasks.jsonl
            num_new_tasks = 0
            src_tasks = src_meta_dir / "tasks.jsonl"
            if src_tasks.exists():
                base_tasks = self.read_jsonl(tasks_out) if tasks_out.exists() else []
                new_tasks = self.read_jsonl(src_tasks)[: actual_episode_counts[i]]
                existing_task_names = {r["task"]: r["task_index"] for r in base_tasks}
                next_idx = max(existing_task_names.values()) + 1 if existing_task_names else 0
                for r_new in new_tasks:
                    if r_new["task"] not in existing_task_names:
                        base_tasks.append({"task": r_new["task"], "task_index": next_idx})
                        existing_task_names[r_new["task"]] = next_idx
                        next_idx += 1
                        num_new_tasks += 1
                self.write_jsonl(sorted(base_tasks, key=lambda x: x["task_index"]), tasks_out)

            # Merge info.json
            src_info = src_meta_dir / "info.json"
            if src_info.exists():
                d_base = json.loads(info_out.read_text()) if info_out.exists() else {}
                d_new = json.loads(src_info.read_text())
                merged_info = d_base.copy()
                # for k in MERGE_NUM_KEYS:
                #     merged_info[k] = merged_info.get(k, 0) + d_new.get(k, 0)
                for k, v in d_new.items():
                    if (
                        k not in MERGE_NUM_KEYS
                        and k
                        not in [
                            "splits",
                            "total_episodes",
                            "total_frames",
                            "chunks_size",
                            "total_chunks",
                            "total_tasks",
                        ]
                        or k == "splits"
                        and not d_base
                    ):
                        merged_info[k] = v
                # total_eps = merged_info.get("total_episodes", 0)
                # merged_info.setdefault("splits", {})["train"] = f"0:{total_eps - 1 if total_eps > 0 else 0}"
                merged_info.setdefault("splits", {})["train"] = "0:100"
                # custom setting for chunk number and chunk size
                # merged_info["chunks_size"] = actual_episode_counts[i]
                merged_info["chunks_size"] = total_episodes_all
                merged_info["total_episodes"] = merged_info.get("total_episodes", 0) + actual_episode_counts[i]
                merged_info["total_frames"] = merged_info.get("total_frames", 0) + actual_frame_counts[i]
                merged_info["total_chunks"] = merged_info.get("total_chunks", 0) + 1
                merged_info["total_tasks"] = merged_info.get("total_tasks", 0) + num_new_tasks

                info_out.write_text(json.dumps(merged_info, indent=2))

            # Merge modality.json
            src_modality = src_meta_dir / "modality.json"
            if src_modality.exists():
                d_base = json.loads(modality_out.read_text()) if modality_out.exists() else {}
                d_new = json.loads(src_modality.read_text())
                merged_modality = d_base.copy()
                for k, v in d_new.items():
                    merged_modality[k] = v
                modality_out.write_text(json.dumps(merged_modality, indent=2))

            current_meta_episode_offset += eps_in_this_ds_for_meta

    def _copy_all_videos_for_merge(
        self,
        dataset_paths: List[Path],
        video_dst_chunk_roots: List[Path],
        chunk_name: str,
        actual_episode_counts: List[int],
        verbose: bool,
    ):
        current_video_start_idx = 0
        for i, (dataset_path, video_dst_chunk_root) in enumerate(zip(dataset_paths, video_dst_chunk_roots)):
            src_video_root = dataset_path / "videos" / chunk_name
            eps_in_this_ds = actual_episode_counts[i]
            if not src_video_root.exists():
                if verbose:
                    print(f"  Video source dir not found: {src_video_root}")
                current_video_start_idx += eps_in_this_ds
                continue

            cam_dirs = sorted(p for p in src_video_root.iterdir() if p.is_dir())
            if not cam_dirs:  # Videos directly under chunk root
                vids_in_chunk = self._natural_sort_paths(src_video_root.glob("episode_*.mp4"))
                for src_vid in vids_in_chunk[:eps_in_this_ds]:
                    dst_idx = self._extract_idx_from_name(src_vid.name) + current_video_start_idx
                    shutil.copy2(src_vid, video_dst_chunk_root / f"episode_{dst_idx:0{PAD}d}.mp4")
            else:  # Videos in camera subdirectories
                for cam_dir_path in cam_dirs:
                    dst_cam_path = video_dst_chunk_root / cam_dir_path.name
                    self.safe_mkdir(dst_cam_path)
                    vids = self._natural_sort_paths(cam_dir_path.glob("episode_*.mp4"))
                    for src_vid_path in vids[:eps_in_this_ds]:
                        dst_idx = self._extract_idx_from_name(src_vid_path.name) + current_video_start_idx
                        shutil.copy2(src_vid_path, dst_cam_path / f"episode_{dst_idx:0{PAD}d}.mp4")
            if verbose:
                print(f"  Copied videos from {dataset_path} with offset {current_video_start_idx}")
            current_video_start_idx += eps_in_this_ds

    # ─────────────────────────────────── DELETE Operation ─────────────────────────────────── #

    def delete_episode_from_dataset(self, ds_dir: Path, ep_id_to_delete: int, chunk_name: str, verbose: bool = False):
        """
        Deletes a specific episode from a dataset and renumbers subsequent episodes.
        Modifies the dataset in-place.
        """
        ds_dir = ds_dir.resolve()
        if not ds_dir.is_dir():
            print(f"Error: Dataset directory not found: {ds_dir}")
            return

        if verbose:
            print(f"Starting delete operation for episode {ep_id_to_delete} in dataset: {ds_dir}")
            print(f"Target chunk: {chunk_name}")

        tgt_stem_to_delete = f"episode_{ep_id_to_delete:0{PAD}d}"
        frames_removed_count = 0
        videos_removed_count = 0  # This will count 1 if any video for the episode is deleted.

        # --- 1. Delete physical files of the target episode ---
        data_chunk_dir = ds_dir / "data" / chunk_name
        tgt_parquet = data_chunk_dir / f"{tgt_stem_to_delete}.parquet"
        if tgt_parquet.exists():
            if verbose:
                print(f"  Deleting Parquet: {tgt_parquet}")
            try:
                df = pd.read_parquet(tgt_parquet)
                frames_removed_count = len(df)
            except Exception as e:
                if verbose:
                    print(f"    Could not read parquet {tgt_parquet} to count frames: {e}")
            tgt_parquet.unlink()

        video_chunk_dir = ds_dir / "videos" / chunk_name
        deleted_any_video_for_ep = False
        if video_chunk_dir.exists():
            camera_subdirs = [d for d in video_chunk_dir.iterdir() if d.is_dir()]
            if camera_subdirs:
                for cam_subdir in camera_subdirs:
                    tgt_video_file = cam_subdir / f"{tgt_stem_to_delete}.mp4"
                    if tgt_video_file.exists():
                        if verbose:
                            print(f"  Deleting video: {tgt_video_file}")
                        tgt_video_file.unlink()
                        deleted_any_video_for_ep = True
            else:
                tgt_video_file = video_chunk_dir / f"{tgt_stem_to_delete}.mp4"
                if tgt_video_file.exists():
                    if verbose:
                        print(f"  Deleting video: {tgt_video_file}")
                    tgt_video_file.unlink()
                    deleted_any_video_for_ep = True
        if deleted_any_video_for_ep:
            videos_removed_count = 1  # Count as 1 episode's worth of videos removed

        images_root_dir = ds_dir / "images"
        if images_root_dir.exists():
            for cam_obs_dir in images_root_dir.iterdir():
                if cam_obs_dir.is_dir():
                    tgt_image_dir = cam_obs_dir / tgt_stem_to_delete
                    if tgt_image_dir.exists() and tgt_image_dir.is_dir():
                        if verbose:
                            print(f"  Deleting image directory: {tgt_image_dir}")
                        shutil.rmtree(tgt_image_dir)

        # --- 2. Shift higher episode files and directories, and patch their content ---
        if data_chunk_dir.exists():
            all_parquets = sorted(
                [p for p in data_chunk_dir.glob("episode_*.parquet") if self._ep_id_from_stem(p.stem) is not None],
                key=lambda p: self._ep_id_from_stem(p.stem),  # type: ignore
            )
            for p_file in all_parquets:
                ep_idx = self._ep_id_from_stem(p_file.stem)
                if ep_idx is not None and ep_idx > ep_id_to_delete:
                    new_idx = ep_idx - 1
                    new_stem = f"episode_{new_idx:0{PAD}d}"
                    new_file_path = p_file.with_name(new_stem + p_file.suffix)
                    if verbose:
                        print(f"  Renaming data {p_file.name} -> {new_file_path.name}")
                    shutil.move(str(p_file), new_file_path)
                    self._patch_parquet_for_delete(new_file_path, -1, verbose)

        if video_chunk_dir.exists():
            video_paths_to_process = []
            camera_subdirs_for_rename = [d for d in video_chunk_dir.iterdir() if d.is_dir()]  # Re-check
            if camera_subdirs_for_rename:
                for cam_subdir in camera_subdirs_for_rename:
                    video_paths_to_process.extend(list(cam_subdir.glob("episode_*.mp4")))
            else:
                video_paths_to_process.extend(list(video_chunk_dir.glob("episode_*.mp4")))

            sorted_video_paths = sorted(
                [p for p in video_paths_to_process if self._ep_id_from_stem(p.stem) is not None],
                key=lambda p: self._ep_id_from_stem(p.stem),  # type: ignore
            )
            for v_file in sorted_video_paths:
                ep_idx = self._ep_id_from_stem(v_file.stem)
                if ep_idx is not None and ep_idx > ep_id_to_delete:
                    new_idx = ep_idx - 1
                    new_stem = f"episode_{new_idx:0{PAD}d}"
                    new_file_path = v_file.with_name(new_stem + v_file.suffix)
                    if verbose:
                        print(f"  Renaming video {v_file.name} -> {new_file_path.name} in {v_file.parent}")
                    shutil.move(str(v_file), new_file_path)

        if images_root_dir.exists():
            for cam_obs_dir in images_root_dir.iterdir():
                if cam_obs_dir.is_dir():
                    image_dirs_to_process = sorted(
                        [
                            d
                            for d in cam_obs_dir.glob("episode_*")
                            if d.is_dir() and self._ep_id_from_stem(d.name) is not None
                        ],
                        key=lambda d: self._ep_id_from_stem(d.name),  # type: ignore
                    )
                    for img_dir in image_dirs_to_process:
                        ep_idx = self._ep_id_from_stem(img_dir.name)
                        if ep_idx is not None and ep_idx > ep_id_to_delete:
                            new_idx = ep_idx - 1
                            new_name = f"episode_{new_idx:0{PAD}d}"
                            new_dir_path = img_dir.with_name(new_name)
                            if verbose:
                                print(f"  Renaming image dir {img_dir.name} -> {new_dir_path.name} in {img_dir.parent}")
                            shutil.move(str(img_dir), new_dir_path)

        # --- 3. Update JSONL metadata files (meta/*.jsonl) ---
        meta_dir = ds_dir / "meta"
        for name in ["episodes_stats.jsonl", "episodes.jsonl", "episodes.json"]:
            path = meta_dir / name
            if path.exists():
                if verbose:
                    print(f"  Updating metadata file: {path.name}")
                self._rewrite_json_or_jsonl_for_delete(path, ep_id_to_delete, -1, verbose)

        # --- 4. Update meta/info.json counts ---
        info_path = meta_dir / "info.json"
        if info_path.exists():
            if verbose:
                print(f"  Updating global metadata: {info_path.name}")
            try:
                meta_info = json.loads(info_path.read_text())
                if isinstance(meta_info.get("total_episodes"), int):
                    meta_info["total_episodes"] = max(0, meta_info["total_episodes"] - 1)
                if isinstance(meta_info.get("total_frames"), int):
                    meta_info["total_frames"] = max(0, meta_info["total_frames"] - frames_removed_count)
                if videos_removed_count > 0 and isinstance(meta_info.get("total_videos"), int):
                    meta_info["total_videos"] = max(0, meta_info["total_videos"] - videos_removed_count)

                if isinstance(meta_info.get("splits"), dict) and isinstance(meta_info["splits"].get("train"), str):
                    start_str, _, end_str = meta_info["splits"]["train"].partition(":")
                    try:
                        start_idx = int(start_str)
                        new_end_idx = int(end_str) - 1
                        if meta_info["total_episodes"] == 0:
                            meta_info["splits"]["train"] = "0:0"  # Or some other indicator of empty
                        elif start_idx <= new_end_idx:
                            meta_info["splits"]["train"] = f"{start_idx}:{new_end_idx}"
                        else:
                            meta_info["splits"]["train"] = f"{start_idx}:{start_idx}"

                    except ValueError:
                        pass
                info_path.write_text(json.dumps(meta_info, indent=2))
            except Exception as e:
                print(f"    Error updating {info_path.name}: {e}")

        print(f"✅ Episode {ep_id_to_delete} deleted and dataset renumbered in {ds_dir}")

    def _patch_parquet_for_delete(self, path: Path, off: int, verbose: bool) -> int:
        """Patches 'episode_index' in a Parquet file by an offset."""
        try:
            df = pd.read_parquet(path)
            nrows = len(df)
            if "episode_index" in df.columns and pd.api.types.is_integer_dtype(df["episode_index"]):
                df["episode_index"] += off  # e.g., off = -1
                df.to_parquet(path, index=False)
            return nrows
        except Exception as e:
            if verbose:
                print(f"    Could not patch Parquet {path}: {e}")
            return 0

    def _rewrite_json_or_jsonl_for_delete(
        self, path: Path, ep_id_to_remove: int, offset_for_shifting: int, verbose: bool
    ):
        """
        Rewrites a JSON or JSONL file:
        - Removes entries matching ep_id_to_remove.
        - Shifts 'episode_index' (and other keys in DELETE_PATCH_KEYS) for entries with episode_index > ep_id_to_remove.
        """
        try:
            content = path.read_text().strip()
            if not content:  # File is empty
                if verbose:
                    print(f"    {path.name} is empty, skipping.")
                return
        except Exception as e:
            if verbose:
                print(f"    Could not read {path.name}: {e}")
            return

        is_json_list_format = content.startswith("[") and content.endswith("]")
        new_data_list_for_output = []  # Stores processed objects for JSON list or strings for JSONL

        original_data_list_parsed = []
        if is_json_list_format:
            try:
                original_data_list_parsed = json.loads(content)
                if not isinstance(original_data_list_parsed, list):  # Should be a list
                    if verbose:
                        print(
                            f"    Content of {path.name} looks like JSON list but is not a list. Processing as JSONL."
                        )
                    is_json_list_format = False  # Fallback
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"    Invalid JSON in {path.name}: {e}. Attempting line-by-line (JSONL).")
                is_json_list_format = False  # Fallback to JSONL

        items_to_process = original_data_list_parsed if is_json_list_format else content.splitlines()

        for item in items_to_process:
            obj_to_process = None
            is_decodable_json = False
            raw_line_if_not_json = item  # For JSONL, item is a line string

            if is_json_list_format:  # item is already a Python object (dict, list, etc.)
                obj_to_process = item
                is_decodable_json = True
            elif isinstance(item, str) and item.strip():  # For JSONL, item is a line string
                try:
                    obj_to_process = json.loads(item)
                    is_decodable_json = True
                except json.JSONDecodeError:
                    is_decodable_json = False  # Keep raw_line_if_not_json

            if is_decodable_json and isinstance(obj_to_process, dict):
                current_ep_idx = obj_to_process.get("episode_index")
                if current_ep_idx == ep_id_to_remove:
                    if verbose:
                        print(f"      Removing episode {ep_id_to_remove} entry from {path.name}")
                    continue  # Skip this episode
                if isinstance(current_ep_idx, int) and current_ep_idx > ep_id_to_remove:
                    obj_to_process = self._patch_indices_recursive_negative(obj_to_process, offset_for_shifting)

            # Add to output list
            if is_json_list_format:
                new_data_list_for_output.append(obj_to_process)
            else:  # JSONL
                if is_decodable_json:  # If it was JSON, serialize it back
                    new_data_list_for_output.append(json.dumps(obj_to_process, separators=(",", ":")))
                else:  # If it was not JSON (e.g. malformed line), keep original line
                    new_data_list_for_output.append(raw_line_if_not_json)

        # Write back to file
        if is_json_list_format:
            path.write_text(json.dumps(new_data_list_for_output, indent=2) + "\n")
        else:  # JSONL
            path.write_text("\n".join(new_data_list_for_output) + ("\n" if new_data_list_for_output else ""))

        if verbose:
            print(f"    {path.name} updated.")

    # ─────────────────────────────────── SPLIT Operation ─────────────────────────────────── #
    def _copy_parquet_and_update_indices_for_split(
        self,
        src_root: Path,
        dst_data_dir: Path,
        task_list: List[str],
        episode_task_list: List[str],
        verbose: bool,
    ) -> int:
        """
        Copy Parquet files and update indices for each task.
        """
        src_chunk_dirs = self._natural_sort_paths((src_root / "data").glob("chunk-*"))
        count_processed = {task: 0 for task in task_list}
        frames_processed = {task: 0 for task in task_list}

        for src_chunk_dir in tqdm(src_chunk_dirs, total=len(src_chunk_dirs), desc="Copying Parquet files", leave=False):
            if not src_chunk_dir.exists():
                if verbose:
                    print(f"Error: Source chunk directory not found: {src_chunk_dir}")
                return 0

            src_files = self._natural_sort_paths(src_chunk_dir.glob("episode_*.parquet"))
            if not src_files:
                if verbose:
                    print(f"Error: No Parquet files found in {src_chunk_dir}")
                return 0

            for src_file_path in tqdm(
                src_files, total=len(src_files), desc=f"Copying Parquet files from {src_chunk_dir.name}", leave=False
            ):
                ep_idx = self._extract_idx_from_name(src_file_path.name)
                task = episode_task_list[ep_idx]
                dst_file_path = (
                    dst_data_dir
                    / f"{slugify(task)}"
                    / "data"
                    / "chunk-000"
                    / f"episode_{count_processed[task]:0{PAD}d}.parquet"
                )
                try:
                    df = pd.read_parquet(src_file_path)
                    if "task_index" in df.columns:
                        df["task_index"] = 0  # reset the index to be 0.
                    if "episode_index" in df.columns:
                        df["episode_index"] = count_processed[task]
                    df.to_parquet(dst_file_path)
                    count_processed[task] += 1
                    frames_processed[task] += len(df)
                except Exception as e:
                    if verbose:
                        print(f"Error: Could not read Parquet file {src_file_path}: {e}")
                    if dst_file_path.exists():
                        dst_file_path.unlink(missing_ok=True)
        return count_processed, frames_processed

    def _split_all_meta_files(
        self,
        dataset_dir: Path,
        output_dir: Path,
        task_list: List[str],
        processed_eps: Dict[str, int],
        processed_frames: Dict[str, int],
        verbose: bool,
    ):
        """
        Split all meta files.
        """

        src_modality = dataset_dir / "meta" / "modality.json"
        src_episodes = dataset_dir / "meta" / "episodes.jsonl"
        src_tasks = dataset_dir / "meta" / "tasks.jsonl"
        src_info = dataset_dir / "meta" / "info.json"

        task_dirs = {task: output_dir / slugify(task) for task in task_list}
        ep_outs = {task: task_dirs[task] / "meta" / "episodes.jsonl" for task in task_list}
        task_outs = {task: task_dirs[task] / "meta" / "tasks.jsonl" for task in task_list}
        info_outs = {task: task_dirs[task] / "meta" / "info.json" for task in task_list}
        modality_outs = {task: task_dirs[task] / "meta" / "modality.json" for task in task_list}

        base_ep_data = self.read_jsonl(src_episodes)
        base_info_data = json.loads(src_info.read_text())
        ep_cnt_per_task = {task: 0 for task in task_list}
        ep_list_per_task = {task: [] for task in task_list}
        if verbose:
            print(f"Splitting dataset {dataset_dir} into {output_dir}")

        for ep_data in base_ep_data:
            task = ep_data["tasks"][0]
            new_ep_data = {
                "episode_index": ep_cnt_per_task[task],
                "tasks": ep_data["tasks"],
                "length": ep_data["length"],
            }
            ep_list_per_task[task].append(new_ep_data)
            ep_cnt_per_task[task] += 1

        for task in task_list:
            # write modality.json
            shutil.copy2(src_modality, modality_outs[task])

            # write tasks.jsonl
            task_json = [{"task_index": 0, "task": task}]
            shutil.copy2(src_tasks, task_outs[task])
            self.write_jsonl(task_json, task_outs[task])

            # write episodes.jsonl
            self.write_jsonl(ep_list_per_task[task], ep_outs[task])

            # write info.json
            new_info_data = base_info_data.copy()
            new_info_data["total_episodes"] = processed_eps[task]
            new_info_data["total_frames"] = processed_frames[task]
            new_info_data["total_tasks"] = 1
            new_info_data["total_chunks"] = 1
            new_info_data["splits"]["train"] = f"0:{ep_cnt_per_task[task]}"
            info_outs[task].write_text(json.dumps(new_info_data, indent=2))

    def _copy_all_videos_for_split(
        self, dataset_dir: Path, output_dir: Path, task_list: List[str], episode_task_list: List[str], verbose: bool
    ):
        """
        Copy all videos for each task.
        """
        ep_to_idx_per_task = {}
        ep_cnt_per_task = {task: 0 for task in task_list}

        for ep_data in self.read_jsonl(dataset_dir / "meta" / "episodes.jsonl"):
            task = ep_data["tasks"][0]
            ep_to_idx_per_task[ep_data["episode_index"]] = ep_cnt_per_task[task]
            ep_cnt_per_task[task] += 1

        src_chunk_dirs = self._natural_sort_paths((dataset_dir / "videos").glob("chunk-*"))
        for src_chunk_dir in tqdm(src_chunk_dirs, total=len(src_chunk_dirs), desc="Copying Videos", leave=False):
            if not src_chunk_dir.exists():
                if verbose:
                    print(f"Error: Source chunk directory not found: {src_chunk_dir}")
                return 0

            cam_dirs = sorted([d for d in src_chunk_dir.iterdir() if d.is_dir()])
            if not cam_dirs:
                vids_in_chunk = self._natural_sort_paths(src_chunk_dir.glob("*.mp4"))
                for src_vid in tqdm(
                    vids_in_chunk,
                    total=len(vids_in_chunk),
                    desc=f"Copying Videos from {src_chunk_dir.name}",
                    leave=False,
                ):
                    ep_idx = self._extract_idx_from_name(src_vid.name)
                    task = episode_task_list[ep_idx]
                    dst_file_path = (
                        output_dir
                        / f"{slugify(task)}"
                        / "videos"
                        / "chunk-000"
                        / f"episode_{ep_to_idx_per_task[ep_idx]:0{PAD}d}.mp4"
                    )
                    self.safe_mkdir(dst_file_path.parent)
                    shutil.copy2(src_vid, dst_file_path)
            else:
                for cam_dir_path in tqdm(
                    cam_dirs, total=len(cam_dirs), desc=f"Copying Videos from {src_chunk_dir.name}", leave=False
                ):
                    vids = self._natural_sort_paths(cam_dir_path.glob("*.mp4"))
                    for src_vid in tqdm(
                        vids, total=len(vids), desc=f"Copying Videos from {cam_dir_path.name}", leave=False
                    ):
                        ep_idx = self._extract_idx_from_name(src_vid.name)
                        task = episode_task_list[ep_idx]
                        dst_file_path = (
                            output_dir
                            / f"{slugify(task)}"
                            / "videos"
                            / "chunk-000"
                            / cam_dir_path.name
                            / f"episode_{ep_to_idx_per_task[ep_idx]:0{PAD}d}.mp4"
                        )
                        self.safe_mkdir(dst_file_path.parent)
                        shutil.copy2(src_vid, dst_file_path)

    def split_dataset(self, dataset_dir: Path, output_dir: Path, verbose: bool = False):
        """
        Split a dataset according to task type.
        """
        if verbose:
            print(f"Splitting dataset {dataset_dir} into {output_dir}")

        task_list = [elem["task"] for elem in self.read_jsonl(dataset_dir / "meta" / "tasks.jsonl")]
        episode_task_list = {
            elem["episode_index"]: elem["tasks"][0] for elem in self.read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        }
        for task in task_list:
            task_output_dir = output_dir / slugify(task)
            meta_dst_dir = task_output_dir / "meta"
            self.safe_mkdir(meta_dst_dir)

            # assume that the dataset is using only one chunk for each task
            self.safe_mkdir(task_output_dir / "data" / "chunk-000")
            self.safe_mkdir(task_output_dir / "videos" / "chunk-000")

        if verbose:
            print("\n--- Processing Parquet Files and Determining Task-Specific Episode Counts ---")
        processed_eps, processed_frames = self._copy_parquet_and_update_indices_for_split(
            dataset_dir, output_dir, task_list, episode_task_list, verbose
        )

        if verbose:
            print("\n--- Processing Metadata Files ---")
        self._split_all_meta_files(dataset_dir, output_dir, task_list, processed_eps, processed_frames, verbose)

        if verbose:
            print("\n--- Processing Videos ---")
        self._copy_all_videos_for_split(dataset_dir, output_dir, task_list, episode_task_list, verbose)

        print(f"\n✅ Split finished!\n  • Output directory: {output_dir}")


"""
Lerobot Dataset Tool - CLI Interface

Example usage:
  python dataset_tool_cli.py merge \\
      --datasets "/path/to/datasetA /path/to/datasetB" \\
      --output_dir /path/to/merged_dataset

  python dataset_tool_cli.py delete \\
      --dataset_dir /path/to/dataset_to_modify \\
      --episode_id 32 \\
      --verbose
"""


def main_cli():
    parser = argparse.ArgumentParser(
        description="Lerobot Dataset Management Tool.",
        formatter_class=argparse.RawTextHelpFormatter,  # To preserve newline in help text
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands. Use <command> -h for details."
    )

    # --- Merge command ---
    parser_merge = subparsers.add_parser(
        "merge",
        help="Merge multiple datasets into one.",
        description=(
            "Merges multiple Lerobot datasets into a new output directory. \n"
            "Episode indices are renumbered, Parquet files are updated, \n"
            "meta files are concatenated/updated, and videos are copied and re-indexed."
        ),
    )
    parser_merge.add_argument(
        "--datasets",
        type=str,
        required=True,
        help='Space-separated string of paths to datasets to merge. e.g., "/path/A /path/B"',
    )
    parser_merge.add_argument(
        "--output_dir", type=Path, required=True, help="Directory where the merged dataset will be saved."
    )
    parser_merge.add_argument(
        "--chunk_name",
        type=str,
        default=CHUNK_NAME_DEFAULT,
        help=f"Name of the data chunk (default: {CHUNK_NAME_DEFAULT}).",
    )
    parser_merge.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to merge. If not provided, all episodes will be merged.",
    )
    parser_merge.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")

    # --- Delete command ---
    parser_delete = subparsers.add_parser(
        "delete",
        help="Delete an episode from a dataset.",
        description=(
            "Deletes a specific episode from a dataset and renumbers all subsequent episodes and their associated files.\n"
            "This operation modifies the dataset IN-PLACE."
        ),
    )
    parser_delete.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the dataset to modify (operation is in-place).",
    )
    parser_delete.add_argument("--episode_id", type=int, required=True, help="ID of the episode to delete.")
    parser_delete.add_argument(
        "--chunk_name",
        type=str,
        default=CHUNK_NAME_DEFAULT,
        help=f"Name of the data chunk (default: {CHUNK_NAME_DEFAULT}).",
    )
    parser_delete.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")

    parser_split = subparsers.add_parser(
        "split",
        help="Split a dataset into multiple tasks.",
        description=("Splits a dataset into multiple tasks. \nThis operation modifies the dataset IN-PLACE."),
    )
    parser_split.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the dataset to modify (operation is in-place).",
    )
    parser_split.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the split dataset will be saved.",
    )
    parser_split.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")

    
    # merge suite
    parser_merge_suite = subparsers.add_parser(
        "merge_suite",
        help="Merge a dataset into task suites.",
        description=("Merges multiple tasks into a single dataset. \nThis operation modifies the dataset IN-PLACE."),
    )
    parser_merge_suite.add_argument(
        "--task_suite_name",
        type=str,
        required=True,
        help=f"Name of the data chunk (default: {CHUNK_NAME_DEFAULT}).",
    )
    parser_merge_suite.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the dataset to modify (operation is in-place).",
    )
    parser_merge_suite.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the dataset to modify (operation is in-place).",
    )
    parser_merge_suite.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()
    manager = DatasetManager()

    if args.command == "merge":
        manager.merge_datasets(args.datasets, args.output_dir, args.chunk_name, args.num_episodes, args.verbose)
    elif args.command == "delete":
        manager.delete_episode_from_dataset(args.dataset_dir, args.episode_id, args.chunk_name, args.verbose)
    elif args.command == "split":
        manager.split_dataset(args.dataset_dir, args.output_dir, args.verbose)
    
    elif args.command == "merge_suite":
            
        # create output dir with task_suite_name
        output_dir = args.output_dir / args.task_suite_name
            
        build_multitask_dataset(
            task_suite_name=args.task_suite_name,
            datasets_root=args.dataset_dir,
            output_dir=output_dir,
            verbose=True,
        )
        
    else:
        parser.print_help()  # Should not be reached due to `required=True` on subparsers


if __name__ == "__main__":
    main_cli()
