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

import copy
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional

import torch
import tyro
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotSingleDataset, LeRobotSubsetDataset, LeRobotMixtureDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, DefaultDataCollator
from gr00t.utils.peft import get_lora_model
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.utils.experiment import CheckpointFormatCallback


import os, json, hashlib
from pathlib import Path

@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

    # Logging parameters
    run_name: str = "default"
    """Run name for logging."""

    # Continual Learning parameters
    cl_type: Literal["none", "ER", "EWC"] = "none"
    """Type of continual learning to use. [none, ER, EWC]"""

    n_memories: int = 1000
    """Number of memories to finetune on for ER (Experience Replay)"""
    
    no_resampling: bool = False
    
    ewc_lambda: float = 50000.0
    """Regularization strength for EWC."""
    
    ewc_gamma: float = 0.9
    """Decay factor for Fisher information accumulation."""

    seed: int = 42


#####################################################################################
# ER helpers
#####################################################################################


class ERDataCollator:
    def __init__(self, replay_loader: DataLoader, base_collator: Optional[callable] = None):
        self.replay_loader = replay_loader
        self._replay_iter = self._cycle(self.replay_loader)
        self.base = base_collator if base_collator is not None else DefaultDataCollator()

    @staticmethod
    def _cycle(dl: DataLoader) -> Iterable:
        while True:
            for batch in dl:
                yield batch

    def __call__(self, current_samples: List[Any]) -> Any:
        replay_samples = next(self._replay_iter)
        combined_samples = current_samples + replay_samples
        return self.base(combined_samples)


#####################################################################################
# main training function
#####################################################################################

saved_indices = {}

def _key(p: str) -> str:
    return os.path.abspath(p)

def _stable_seed_for(p: str, base_seed: int) -> int:
    s = f"{_key(p)}::{base_seed}"
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)

def construct_dataset(config: ArgsConfig):
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    replay_dataset = None

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=copy.deepcopy(transforms),
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        if config.cl_type == "ER":
            
            if config.no_resampling:
                
                # All previous tasks
                truncated_prev = []
                n_tasks = len(config.dataset_path)
                for idx, p in enumerate(config.dataset_path): # all including the current one
                    
                    print(f"Loading previous task {idx} dataset from {p}")
                    assert os.path.exists(p), f"Dataset path {p} does not exist"
                    ds = LeRobotSingleDataset(
                        dataset_path=p,
                        modality_configs=modality_configs,
                        transforms=copy.deepcopy(transforms),
                        embodiment_tag=embodiment_tag,
                        video_backend=config.video_backend,
                    )
                    
                    if config.n_memories == -1:
                        truncated_prev.append(ds)
                    else:
                        # previous tasks → truncate to n_memories and add to REPLAY
                        buffer_size = min(config.n_memories, len(ds))
                        if p in saved_indices:
                            indices = saved_indices[p]
                        else:
                            g = torch.Generator()
                            g.manual_seed(_stable_seed_for(p, config.seed))
                            indices = torch.randperm(len(ds), generator=g)[:buffer_size].tolist()                                     
                            # indices = torch.randperm(len(ds))[:buffer_size].tolist()
                            saved_indices[p] = indices
                        truncated_prev.append(LeRobotSubsetDataset(ds, indices=indices))
                        
                if len(truncated_prev) > 0:
                    # train_dataset = ConcatDataset(truncated_prev)
                    
                    mixture_pairs = [
                        (dataset, 1.0)  # we will use equal weights for all datasets
                        for dataset in truncated_prev
                    ]
                    
                    train_dataset = LeRobotMixtureDataset(
                        data_mixture=mixture_pairs,              
                        mode="train",   
                        seed=config.seed,
                    )
                    
                replay_dataset = None # train dataset is the mixed one, no replay dataset
                
            else:
                
                # All previous tasks
                truncated_prev = []
                n_tasks = len(config.dataset_path)
                for idx, p in enumerate(config.dataset_path[:-1]):
                    
                    print(f"Loading previous task {idx} dataset from {p}")
                    assert os.path.exists(p), f"Dataset path {p} does not exist"
                    ds = LeRobotSingleDataset(
                        dataset_path=p,
                        modality_configs=modality_configs,
                        transforms=copy.deepcopy(transforms),
                        embodiment_tag=embodiment_tag,
                        video_backend=config.video_backend,
                    )

                    if config.n_memories == -1:
                        truncated_prev.append(ds)
                    else:
                        # previous tasks → truncate to n_memories and add to REPLAY
                        buffer_size = min(config.n_memories, len(ds))
                        print(ds.dataset_name)
                        if p in saved_indices:
                            print("reach p in saved_indices")
                            indices = saved_indices[p]
                        else:
                            print("reach p NOT in saved_indices")
                            g = torch.Generator()
                            g.manual_seed(_stable_seed_for(p, config.seed))
                            indices = torch.randperm(len(ds), generator=g)[:buffer_size].tolist()
                            # indices = torch.randperm(len(ds))[:buffer_size].tolist()
                            saved_indices[p] = indices
                            
                        print(indices)
                        # indices = torch.randperm(len(ds))[:buffer_size].tolist()
                        truncated_prev.append(LeRobotSubsetDataset(ds, indices=indices))
                        
                if len(truncated_prev) > 0:
                    replay_dataset = ConcatDataset(truncated_prev)
                
                # The current task
                current_path = config.dataset_path[-1]
                train_dataset = LeRobotSingleDataset(
                    dataset_path=current_path,
                    modality_configs=modality_configs,
                    transforms=transforms,
                    embodiment_tag=embodiment_tag,
                    video_backend=config.video_backend,
                )
                print("++++++++++++++++++++++++++++++++++")
            
        else:
            
            current_path = config.dataset_path[-1]
            train_dataset = LeRobotSingleDataset(
                dataset_path=current_path,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )

            replay_dataset = None

    return train_dataset, replay_dataset


def single_task_loop(config: ArgsConfig, ewc_fisher=None, ewc_checkpoint=None):
    """Main training function."""
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    train_dataset, replay_dataset = construct_dataset(config)

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    # Update action_horizon to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    if data_action_horizon != model.action_head.config.action_horizon:
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {model.action_head.config.action_horizon})"
        )

        # Update the action head config
        new_action_head_config = model.action_head.config
        new_action_head_config.action_horizon = data_action_horizon

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=8,
        report_to=config.report_to,
        seed=config.seed,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # ------------ ER replay loader + collator ------------
    custom_collator = None
    if config.cl_type == "ER" and replay_dataset is not None:
        # Build a replay loader that yields batches with the SAME collate as Trainer (DefaultDataCollator)
        base_collator = DefaultDataCollator()

        def _identity_collate(batch):
            # return the list unchanged
            return batch

        replay_loader = DataLoader(  # TODO: NEED TO CHECK
            replay_dataset,
            batch_size=training_args.per_device_train_batch_size,
            sampler=RandomSampler(replay_dataset),
            num_workers=0,  # config.dataloader_num_workers,
            pin_memory=False,
            persistent_workers=False,  # config.dataloader_num_workers > 0,
            # collate_fn=base_collator,
            collate_fn=_identity_collate,
            drop_last=False,
        )
        custom_collator = ERDataCollator(replay_loader=replay_loader, base_collator=base_collator)
        print("[ER] Using ERDataCollator: each step merges current + replay batches.")

    # the below also include case where config.cl_type == "ER" and replay_dataset is None
    
    # 2.2 run experiment
    if config.cl_type == "EWC":
        experiment = EWCTrainRunner(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            resume_from_checkpoint=config.resume,
            data_collator=custom_collator,
            ewc_fisher=ewc_fisher,
            ewc_checkpoint=ewc_checkpoint,
            ewc_lambda=config.ewc_lambda,
        )
    else:
        experiment = TrainRunner(
            train_dataset=train_dataset,
            model=model,
            training_args=training_args,
            resume_from_checkpoint=config.resume,
            data_collator=custom_collator,
        )

    # 2.3 run experiment
    experiment.train()

    current_fisher = None
    current_params = None

    if config.cl_type == "EWC":
         # Compute fisher on the training dataset of the current task
         # Note: EWC typically computes fisher on the *task just finished* to protect it.
         # So we compute fisher for task_i after training task_i.
         current_fisher = compute_fisher(experiment.trainer.model, train_dataset, device=experiment.trainer.args.device)
         
         # Capture current parameters
         current_params = {n: p.data.clone() for n, p in experiment.trainer.model.named_parameters() if p.requires_grad}
         
    return current_fisher, current_params


def compute_fisher(model, dataset, device, num_workers=8):
    """
    Compute Fisher Information Matrix (diagonal approximation) for EWC.
    Returns a dictionary {param_name: fisher_value_tensor}.
    """
    model.eval()
    fisher = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            fisher[n] = torch.zeros_like(p.data)

    # Use a small batch size for fisher computation to avoid OOM
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=DefaultDataCollator(),
    )

    print("Computing Fisher Information...")
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        model.zero_grad()
        # Forward pass to get loss
        # We use the negative log likelihood (loss)
        outputs = model(batch)
        loss = outputs["loss"]
        
        # Backward pass to get gradients
        loss.backward()
        
        # Accumulate squared gradients
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2
    
    # Average over the number of batches (or samples? user snippet used len(dataloader))
    # User snippet: fish /= len(dataloader) -> this is average over batches.
    # To be precise, it should be average over samples, but if batch_size is constant, len(dataloader) is fine.
    # Let's stick to len(dataloader) to match magnitude roughly.
    num_batches = len(dataloader)
    if num_batches > 0:
        for n in fisher:
            fisher[n] /= num_batches
            
    return fisher


class EWCDualBrainTrainer(DualBrainTrainer):
    def __init__(self, ewc_fisher=None, ewc_checkpoint=None, ewc_lambda=0.0, **kwargs):
        self.ewc_fisher = ewc_fisher
        self.ewc_checkpoint = ewc_checkpoint
        self.ewc_lambda = ewc_lambda
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Determine if we should piggyback on parent's compute_loss
        # DualBrainTrainer compute_loss returns (loss, outputs) or loss
        
        outputs = model(inputs)
        loss = outputs["loss"]
        
        # Add EWC penalty
        if self.ewc_fisher is not None and self.ewc_checkpoint is not None:
            ewc_loss = 0.0
            # We iterate over the model parameters
            for n, p in model.named_parameters():
                if n in self.ewc_fisher and n in self.ewc_checkpoint:
                    # Retrieve fisher and old params, move to current device
                    fish = self.ewc_fisher[n].to(p.device)
                    old_p = self.ewc_checkpoint[n].to(p.device)
                    
                    # Compute penalty: sum(F_i * (theta_i - theta_i_old)^2)
                    ewc_loss += (fish * (p - old_p) ** 2).sum()
            
            loss += self.ewc_lambda * ewc_loss
            
            # Optional: Log the EWC loss component? 
            # We can't easily log here without access to self.log or changing returns significantly 
            # effectively. But since we are just adding to scalar loss, it's fine.
            
        return (loss, outputs) if return_outputs else loss


class EWCTrainRunner(TrainRunner):
    def __init__(
        self,
        model: GR00T_N1_5,
        training_args: TrainingArguments,
        train_dataset: LeRobotSingleDataset | LeRobotMixtureDataset,
        resume_from_checkpoint: bool = False,
        data_collator=None,
        ewc_fisher=None,
        ewc_checkpoint=None,
        ewc_lambda=0.0,
    ):
        self.ewc_fisher = ewc_fisher
        self.ewc_checkpoint = ewc_checkpoint
        self.ewc_lambda = ewc_lambda
        super().__init__(model, training_args, train_dataset, resume_from_checkpoint, data_collator)

    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        global_batch_size=None,
    ):
        # Set the gradient accumulation steps if global_batch_size is provided
        if global_batch_size is not None:
            bs = training_args.per_device_train_batch_size
            num_gpus = torch.cuda.device_count()
            grad_acc = max(1, global_batch_size // (bs * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_batch_size}, set gradient accumulation steps to {grad_acc}"
            )

        # Create the EWC trainer
        trainer = EWCDualBrainTrainer(
            ewc_fisher=self.ewc_fisher,
            ewc_checkpoint=self.ewc_checkpoint,
            ewc_lambda=self.ewc_lambda,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )

        # Add checkpoint format callback to ensure experiment_cfg is copied to each checkpoint
        run_name = training_args.run_name
        ckpt_format_callback = CheckpointFormatCallback(
            run_name=run_name, exp_cfg_dir=self.exp_cfg_dir
        )
        trainer.add_callback(ckpt_format_callback)

        # Log dataloader information
        # Note: trainer.get_train_dataloader() initializes the dataloader
        try:
             train_dl_len = len(trainer.get_train_dataloader())
             print(
                f"train dataloader length: {train_dl_len}\n"
                f"train dataset length: {len(trainer.train_dataset)}\n"
                f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
                flush=True,
            )
        except Exception as e:
            print(f"Could not get dataloader info: {e}")

        return trainer


def main(config: ArgsConfig):
    num_tasks = len(config.dataset_path)
    task_names = [Path(p).name for p in config.dataset_path]
    ewc_fisher = {}
    ewc_checkpoint = {}

    for task_id in range(num_tasks):
        new_config = copy.deepcopy(config)
        new_config.run_name = f"{config.run_name}_task_{'-'.join([str(i) for i in range(task_id + 1)]) if config.cl_type == 'ER' else str(task_id)}"
        new_config.dataset_path = (
            config.dataset_path[: task_id + 1] if config.cl_type == "ER" else [config.dataset_path[task_id]]
        )
        new_config.output_dir = os.path.join(config.output_dir, f"task_{task_id}")
        if task_id > 0:
            new_config.base_model_path = os.path.join(
                config.output_dir, f"task_{task_id - 1}", f"checkpoint-{config.max_steps}"
            )
        
        # Pass ewc args
        new_fisher, new_params = single_task_loop(new_config, ewc_fisher=ewc_fisher, ewc_checkpoint=ewc_checkpoint)
        
        if config.cl_type == "EWC" and new_fisher is not None:
             print(f"Updating EWC Fisher and Checkpoint after task {task_id}")
             # Update Checkpoint to the latest model
             ewc_checkpoint = new_params
             
             # Accumulate Fisher
             if not ewc_fisher:
                 ewc_fisher = new_fisher
             else:
                 for n in ewc_fisher:
                     if n in new_fisher:
                        # Move to same device (CPU usually for storage) if necessary, but here likely all on GPU or same device
                        # Assuming consistent device usage.
                        ewc_fisher[n] *= config.ewc_gamma
                        ewc_fisher[n] += new_fisher[n]


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert config.num_gpus <= available_gpus, (
        f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    )
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
