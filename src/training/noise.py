"""Exploration noise generation for reinforcement learning.

This module provides noise generation classes for exploration in continuous
action spaces, particularly the Ornstein-Uhlenbeck process.
"""

import numpy as np
import torch as T

from config import NoiseConfig


class OUActionNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Generates noise that is correlated in time, which is beneficial for
    continuous control tasks with inertia.

    Args:
        config: NoiseConfig containing mu, sigma, theta, dt, x0, action_dimensions
    """

    def __init__(self, config: NoiseConfig) -> None:
        self.sigma = config.sigma
        self.theta = config.theta
        self.dt = config.dt
        self.x0 = config.x0
        self.action_dimensions = config.action_dimensions

        # Convert mu to numpy array
        if isinstance(config.mu, (int, float)):
            self.mu = np.array([config.mu] * config.action_dimensions)
        else:
            self.mu = np.array(config.mu)

        # Initialize noise state
        self.noise: np.ndarray = np.zeros(config.action_dimensions)
        self.reset()

    def reset(self) -> None:
        """Reset noise state to initial value."""
        if self.x0 is None:
            self.noise = self.mu.copy()
        else:
            self.noise = np.ones(self.action_dimensions) * self.x0

    def generate(self) -> T.Tensor:
        """Generate noise for the current step.

        Updates the internal noise state using the OU process and returns
        the new noise value.

        Returns:
            Noise tensor of shape (action_dimensions,)
        """
        random_noise = np.random.normal(size=(self.action_dimensions,))
        self.noise = (
            self.noise
            + self.theta * (self.mu - self.noise) * self.dt
            + self.sigma * np.sqrt(self.dt) * random_noise
        )
        return T.from_numpy(self.noise.copy()).float()
