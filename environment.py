"""Environment management and reward shaping for Lunar Lander."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Tuple, Set, Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv

from config import RunConfig, EnvironmentConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentBundle:
    """Bundle of environments for training.

    Attributes:
        vec_env: Vectorized environment for parallel training
        render_env: Single environment for rendered episodes
        render_episodes: Set of episode numbers to render
    """

    vec_env: SyncVectorEnv
    render_env: gym.Env
    render_episodes: Set[int]


@contextmanager
def create_environments(
    run_config: RunConfig,
    env_config: EnvironmentConfig
) -> Generator[EnvironmentBundle, None, None]:
    """Context manager for creating and managing Gymnasium environments.

    Ensures proper cleanup of environments even if an exception occurs.

    Args:
        run_config: Configuration for training runs
        env_config: Environment configuration

    Yields:
        EnvironmentBundle containing vectorized and render environments
    """
    # Create vectorized environment for fast parallel training
    vec_env = SyncVectorEnv([
        lambda: gym.make(env_config.env_name)
        for _ in range(run_config.num_envs)
    ])

    # Create single environment for rendered episodes
    render_env = gym.make(env_config.env_name, render_mode="human")
    render_env.metadata["render_fps"] = run_config.framerate

    logger.info(f"Created vectorized env with {run_config.num_envs} parallel environments")

    bundle = EnvironmentBundle(
        vec_env=vec_env,
        render_env=render_env,
        render_episodes=set(run_config.render_episodes)
    )

    try:
        yield bundle
    finally:
        vec_env.close()
        render_env.close()
        logger.info("Environments closed")


def shape_reward(
    state: np.ndarray,
    base_reward: float,
    done: bool
) -> float:
    """Apply reward shaping to provide intermediate learning signals.

    LunarLander state format:
    [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]

    Shaping rewards:
    - Bonus for being low (y_pos < 0.25): encourages descent
    - Bonus for single leg contact: encourages landing attempt
    - Larger bonus for both legs contact: encourages stable landing
    - Small bonus for downward velocity: discourages hovering

    Args:
        state: Current state observation
        base_reward: Original reward from environment
        done: Whether episode has terminated

    Returns:
        Shaped reward value
    """
    shaped_reward = base_reward

    y_pos = state[1]
    y_vel = state[3]
    leg1_contact = state[6]
    leg2_contact = state[7]

    # Reward for being close to ground
    if y_pos < 0.25:
        shaped_reward += 2

    # Reward for leg contact (encourages landing attempts)
    if (leg1_contact and not leg2_contact) or (not leg1_contact and leg2_contact):
        shaped_reward += 10

    # Larger reward for stable landing (both legs)
    if leg1_contact and leg2_contact:
        shaped_reward += 20

    # Small reward for descending (discourages hovering)
    if y_vel < -0.05:
        shaped_reward += 1

    return shaped_reward


def compute_noise_scale(
    episode: int,
    initial_scale: float,
    final_scale: float,
    decay_episodes: int
) -> float:
    """Compute the exploration noise scale for a given episode.

    Linearly decays from initial_scale to final_scale over decay_episodes.

    Args:
        episode: Current episode number
        initial_scale: Starting noise scale
        final_scale: Minimum noise scale
        decay_episodes: Number of episodes to decay over

    Returns:
        Current noise scale
    """
    if episode >= decay_episodes:
        return final_scale

    progress = episode / decay_episodes
    return initial_scale - (initial_scale - final_scale) * progress


class EpisodeManager:
    """Manages per-environment episode state for vectorized training.

    Tracks rewards, actions, and shaped bonuses for each parallel environment.

    Args:
        num_envs: Number of parallel environments
    """

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.reset_all()

    def reset_all(self) -> None:
        """Reset tracking for all environments."""
        self.rewards: list[list[float]] = [[] for _ in range(self.num_envs)]
        self.shaped_bonuses: list[float] = [0.0 for _ in range(self.num_envs)]
        self.actions: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]

    def reset_env(self, env_idx: int) -> None:
        """Reset tracking for a single environment.

        Args:
            env_idx: Index of the environment to reset
        """
        self.rewards[env_idx] = []
        self.shaped_bonuses[env_idx] = 0.0
        self.actions[env_idx] = []

    def add_step(
        self,
        env_idx: int,
        reward: float,
        shaped_reward: float,
        action: np.ndarray
    ) -> None:
        """Record a step for an environment.

        Args:
            env_idx: Index of the environment
            reward: Original environment reward
            shaped_reward: Shaped reward value
            action: Action taken
        """
        self.rewards[env_idx].append(reward)
        self.shaped_bonuses[env_idx] += (shaped_reward - reward)
        self.actions[env_idx].append(action)

    def get_episode_stats(
        self,
        env_idx: int
    ) -> Tuple[float, float, float, np.ndarray]:
        """Get episode statistics for a completed environment.

        Args:
            env_idx: Index of the environment

        Returns:
            Tuple of (total_reward, env_reward, shaped_bonus, actions_array)
        """
        env_reward = float(np.sum(self.rewards[env_idx]))
        shaped_bonus = self.shaped_bonuses[env_idx]
        total_reward = env_reward + shaped_bonus
        actions_array = np.array(self.actions[env_idx])

        return total_reward, env_reward, shaped_bonus, actions_array
