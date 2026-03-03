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
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import (
    CategorySpecificLinear,
    CategorySpecificMLP,
    swish,
)
# from gr00t.model.action_head.cross_attention_dit import SelfAttentionTransformer


class BroNet(torch.nn.Module):
    def __init__(
        self, input_dim: int, hidden_size: int, depth: int, add_final_layer: bool = False, output_dim: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.add_final_layer = add_final_layer
        self.output_dim = output_dim

        # Create the residual blocks based on depth
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.residual_blocks = torch.nn.ModuleList()
        for _ in range(depth):
            block = ResidualBlock(hidden_size)
            self.residual_blocks.append(block)

        if add_final_layer:
            self.final_layer = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layernorm(self.input_projection(x)))
        # Pass through each residual block sequentially
        for block in self.residual_blocks:
            x = block(x)

        if self.add_final_layer:
            x = self.final_layer(x)

        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            # First dense block
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            # Second dense block
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            # Final transformation before residual connection
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block(x)
        return x + identity


class DoubleCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, depth: int, add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.Q1 = BroNet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )
        self.Q2 = BroNet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )

    def forward(self, states, actions):
        B = states.shape[0]
        state_action = torch.cat([states.reshape(B, -1), actions.reshape(B, -1)], axis=1)
        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        return q1, q2


class Value(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, depth: int, add_final_layer: bool = True, output_dim: int = 1):
        super().__init__()
        self.value = BroNet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            add_final_layer=add_final_layer,
            output_dim=output_dim,
        )

    def forward(self, states):
        B = states.shape[0]
        v = self.value(states.reshape(B, -1))
        return v


class MultiEmbodimentActionCriticEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)

    def forward(self, actions, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Standard action MLP step for shape => (B, T, w)
        a_emb = swish(self.W1(actions, cat_ids))

        # 2) W2 => (B, T, w)
        x = self.W2(a_emb, cat_ids)

        return x


@dataclass
class CriticConfig(PretrainedConfig):
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding dimension."})
    # backbone_embedding_dim: int = field(default=1536, metadata={"help": "Backbone embedding dimension."})
    hidden_size: int = field(default=1024, metadata={"help": "Hidden dimension."})
    depth: int = field(default=2, metadata={"help": "Depth of the network."})
    add_final_layer: bool = field(default=True, metadata={"help": "Whether to add a final layer."})
    output_dim: int = field(default=1, metadata={"help": "Output dimension."})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    max_state_dim: int = field(default=None, metadata={"help": "Maximum state dimension."})

    # RL parameters
    expectile: float = field(default=0.9, metadata={"help": "Expectile for expectile loss."})
    q_agg: str = field(default="min", metadata={"help": "Aggregation function for critic loss."})
    discount: float = field(default=0.999, metadata={"help": "Discount factor for MDP."})
    nstep: int = field(default=1, metadata={"help": "Number of steps for reward."})
    normalize_q: bool = field(default=False, metadata={"help": "Whether to normalize the Q-value."})
    alpha: float = field(default=10.0, metadata={"help": "Alpha for actor loss."})
    tau: float = field(default=0.005, metadata={"help": "Tau for target critic update."})

    # # VLLN parameters
    # use_vlln: bool = field(default=True, metadata={"help": "Whether to use VLLN."})
    # vl_self_attention_cfg: dict = field(default=None, metadata={"help": "VLLN self attention configuration."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class Critic(nn.Module):
    config_class = CriticConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: CriticConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.critic_action_encoder = MultiEmbodimentActionCriticEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        # self.vlln = (
        #     nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        # )
        # self.vl_self_attention = (
        #     SelfAttentionTransformer(**config.vl_self_attention_cfg)
        #     if config.use_vlln
        #     else nn.Identity()
        # )

        self.value = Value(
            input_dim=self.input_embedding_dim,
            hidden_size=config.hidden_size,
            depth=config.depth,
            add_final_layer=config.add_final_layer,
            output_dim=config.output_dim,
        )

        self.critic = DoubleCritic(
            input_dim=self.input_embedding_dim * (config.action_horizon + 1),
            hidden_size=config.hidden_size,
            depth=config.depth,
            add_final_layer=config.add_final_layer,
            output_dim=config.output_dim,
        )

        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.config = config

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return torch.mean(weight * (diff**2))

    def set_trainable_parameters(self, tune_projector: bool):
        self.tune_projector = tune_projector
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not self.tune_projector:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    # def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
    #     backbone_features = backbone_output["backbone_features"]
    #     backbone_features = self.vlln(backbone_features)
    #     backbone_features = self.vl_self_attention(backbone_features)
    #     backbone_output["backbone_features"] = backbone_features
    #     return backbone_output

    def forward(self, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # backbone_output = self.process_backbone_output(backbone_output)

        # vl_embeds = backbone_output.backbone_features

        # Get vision and language embeddings.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        action_critic_features = self.critic_action_encoder(action_input.action, embodiment_id)

        # Critic loss 1) value loss
        with torch.no_grad():
            q1, q2 = self.target_critic(state_features, action_critic_features)
            if self.config.q_agg == "min":
                q = torch.minimum(q1, q2)
            elif self.config.q_agg == "mean":
                q = (q1 + q2) / 2

        v = self.value(state_features)
        value_loss = self.expectile_loss(q - v, q - v, self.config.expectile)

        # Critic loss 2) critic loss
        next_state_features = self.state_encoder(action_input.next_state, embodiment_id)
        with torch.no_grad():
            next_v = self.value(next_state_features)
            q = (
                action_input.reward
                + (self.config.discount ** (self.config.nstep * self.config.action_horizon))
                * action_input.done
                * next_v
            )
        q1, q2 = self.critic(state_features, action_critic_features)
        critic_loss = ((q - q1) ** 2 + (q - q2) ** 2).mean()

        total_loss = critic_loss + value_loss

        output_dict = {
            "loss": total_loss,
            "critic_loss": critic_loss,
            "value_loss": value_loss,
        }
        return BatchFeature(data=output_dict)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
