"""Exploration noise generation for reinforcement learning.

This module provides noise generation classes for exploration in continuous
action spaces, particularly the Ornstein-Uhlenbeck process.
"""

from typing import Optional

import numpy as np
import torch as T

from config import NoiseConfig


class OUActionNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Generates noise that is correlated in time, which is beneficial for
    continuous control tasks with inertia.

    Args:
        config: NoiseConfig containing mu, sigma, theta, dt, x0, action_dimensions
        num_envs: Number of parallel environments to generate noise for
    """

    def __init__(self, config: NoiseConfig, num_envs: int = 1) -> None:
        self.sigma = config.sigma
        self.theta = config.theta
        self.dt = config.dt
        self.x0 = config.x0
        self.action_dimensions = config.action_dimensions
        self.num_envs = num_envs

        # Convert mu to numpy array
        if isinstance(config.mu, (int, float)):
            self.mu = np.array([config.mu] * config.action_dimensions)
        else:
            self.mu = np.array(config.mu)

        # Initialize noise state
        self.noise: np.ndarray = np.zeros((num_envs, config.action_dimensions))
        self.reset()

    def reset(self, env_idx: Optional[int] = None) -> None:
        """Reset noise state to initial value.

        Args:
            env_idx: If provided, reset only that environment's noise.
                     If None, reset all environments.
        """
        if env_idx is not None:
            if self.x0 is None:
                self.noise[env_idx] = self.mu.copy()
            else:
                self.noise[env_idx] = np.ones(self.action_dimensions) * self.x0
        else:
            if self.x0 is None:
                self.noise = np.tile(self.mu, (self.num_envs, 1))
            else:
                self.noise = np.ones((self.num_envs, self.action_dimensions)) * self.x0

    def generate(self) -> T.Tensor:
        """Generate noise for all environments.

        Returns:
            Noise tensor of shape (num_envs, action_dimensions)
        """
        random_noise = np.random.normal(size=(self.num_envs, self.action_dimensions))
        self.noise = (
            self.noise
            + self.theta * (self.mu - self.noise) * self.dt
            + self.sigma * np.sqrt(self.dt) * random_noise
        )
        return T.from_numpy(self.noise).float()

    def generate_single(self, env_idx: int = 0) -> T.Tensor:
        """Generate noise for a single environment.

        Args:
            env_idx: Index of the environment (default: 0)

        Returns:
            Noise tensor of shape (action_dimensions,)
        """
        random_noise = np.random.normal(size=(1, self.action_dimensions))
        single_noise = (
            self.noise[env_idx:env_idx + 1]
            + self.theta * (self.mu - self.noise[env_idx:env_idx + 1]) * self.dt
            + self.sigma * np.sqrt(self.dt) * random_noise
        )
        return T.from_numpy(single_noise[0]).float()
