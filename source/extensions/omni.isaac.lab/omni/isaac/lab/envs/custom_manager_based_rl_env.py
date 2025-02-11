# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from omni.isaac.version import get_version

from omni.isaac.lab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from .manager_based_rl_env import ManagerBasedRLEnv
from .custom_manager_based_rl_env_cfg import CustomManagerBasedRLEnvCfg


class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    """The superclass for the manager-based workflow reinforcement learning-based environments."""


    def __init__(self, cfg: CustomManagerBasedRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.num_clutter_objects = cfg.num_clutter_objects
        self.position_dim = 3
        self.adversary_action = torch.zeros((self.num_envs, self.num_clutter_objects * self.position_dim)).to(self.device)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        super()._reset_idx(env_ids)
        self.adversarial_reset(env_ids)

    def adversarial_reset(
        self, reset_env_ids: Sequence[int]
    ) -> tuple[VecEnvObs, dict]:
        """Reset the environment.

        Returns:
            np.ndarray: The initial observation.
        """
        positions = self.adversary_action[reset_env_ids]
        curr_state = self.scene.get_state(False) # curr_state is full sim state
        for i in range(self.num_clutter_objects):
            pose = curr_state["rigid_object"][f"clutter_object{i+1}"]['root_pose'][reset_env_ids]
            val1 = positions[:, i*3] * 0.1
            val2 = positions[:, i*3+1] * 0.2
            val3 = positions[:, i*3+2] * 0.025 + 0.15
            pose[:, :3] += torch.stack([val1, val2, val3], dim=-1).to(pose.device)
        self.scene.reset_to(curr_state, reset_env_ids)

