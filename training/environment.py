"""Environment management and reward shaping for Lunar Lander."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Tuple, Set, Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv

from config import RunConfig, EnvironmentConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentBundle:
    """Bundle of environments for training.

    Attributes:
        vec_env: Vectorized environment for parallel training (async for speed)
        render_env: Single environment for rendered episodes
        render_episodes: Set of episode numbers to render
    """

    vec_env: AsyncVectorEnv
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
    # AsyncVectorEnv runs environments in separate processes for true parallelization
    vec_env = AsyncVectorEnv([
        lambda: gym.make(env_config.env_name)
        for _ in range(run_config.num_envs)
    ])

    # Create single environment for rendered episodes
    # Use rgb_array mode so we control rendering and can add overlays without flicker
    render_env = gym.make(env_config.env_name, render_mode="rgb_array")
    render_env.metadata["render_fps"] = run_config.framerate

    logger.info(f"Created async vectorized env with {run_config.num_envs} parallel processes")

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
    terminated: bool,
    step: int = 0
) -> float:
    """Apply reward shaping to provide intermediate learning signals.

    LunarLander state format:
    [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]

    Shaping rewards (gated on descending to prevent hovering):
    - Time penalty: -0.05 per step (discourages hovering/long episodes)
    - Bonus for being low (y_pos < 0.25): only if descending
    - Bonus for leg contact: only if descending
    - Stability bonus: only if descending
    - Terminal landing bonus: +100 for successful landing with both legs

    Args:
        state: Current state observation
        base_reward: Original reward from environment
        terminated: Whether episode has terminated (for landing bonus)
        step: Current step number in episode (unused, kept for API compatibility)

    Returns:
        Shaped reward value
    """
    shaped_reward = base_reward

    y_pos = state[1]
    y_vel = state[3]
    angle = state[4]
    leg1_contact = state[6]
    leg2_contact = state[7]

    # Time penalty to discourage hovering (-0.05 per step)
    shaped_reward -= 0.05

    # All per-step bonuses ONLY apply if descending (prevents hover exploitation)
    is_descending = y_vel < -0.05

    if is_descending:
        # Reward for being close to ground
        if y_pos < 0.25:
            shaped_reward += 0.5

        # Reward for leg contact
        if (leg1_contact and not leg2_contact) or (not leg1_contact and leg2_contact):
            shaped_reward += 2

        # Reward for both legs contact
        if leg1_contact and leg2_contact:
            shaped_reward += 5

        # Stability bonus for staying upright (angle near 0)
        if abs(angle) < 0.1:
            shaped_reward += 0.3
        elif abs(angle) < 0.2:
            shaped_reward += 0.1

    # Terminal landing bonus: big reward for successful landing
    # Successful = terminated with both legs on ground (not a crash)
    # Crashes give -100 from env, landings give +100, so base_reward > 0 means landed
    if terminated and leg1_contact and leg2_contact and base_reward > 0:
        shaped_reward += 100

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
    Uses pre-allocated arrays for better performance.

    Args:
        num_envs: Number of parallel environments
        max_episode_length: Maximum expected episode length (default 1000 for LunarLander)
        state_dim: Dimension of observation space (default 8 for LunarLander)
        action_dim: Dimension of action space (default 2 for LunarLander)
    """

    def __init__(
        self,
        num_envs: int,
        max_episode_length: int = 1000,
        state_dim: int = 8,
        action_dim: int = 2
    ) -> None:
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset_all()

    def reset_all(self) -> None:
        """Reset tracking for all environments."""
        # Pre-allocate arrays for better performance
        self.rewards = np.zeros((self.num_envs, self.max_episode_length), dtype=np.float32)
        self.actions = np.zeros((self.num_envs, self.max_episode_length, self.action_dim), dtype=np.float32)
        self.observations = np.zeros((self.num_envs, self.max_episode_length, self.state_dim), dtype=np.float32)
        self.shaped_bonuses = np.zeros(self.num_envs, dtype=np.float32)
        self.step_counts = np.zeros(self.num_envs, dtype=np.int32)

    def reset_env(self, env_idx: int) -> None:
        """Reset tracking for a single environment.

        Args:
            env_idx: Index of the environment to reset
        """
        self.step_counts[env_idx] = 0
        self.shaped_bonuses[env_idx] = 0.0

    def add_step(
        self,
        env_idx: int,
        reward: float,
        shaped_reward: float,
        action: np.ndarray,
        observation: np.ndarray
    ) -> None:
        """Record a step for an environment.

        Args:
            env_idx: Index of the environment
            reward: Original environment reward
            shaped_reward: Shaped reward value
            action: Action taken
            observation: State observation at this step
        """
        step = self.step_counts[env_idx]
        if step < self.max_episode_length:
            self.rewards[env_idx, step] = reward
            self.actions[env_idx, step] = action
            self.observations[env_idx, step] = observation
            self.shaped_bonuses[env_idx] += (shaped_reward - reward)
            self.step_counts[env_idx] += 1

    def get_episode_stats(
        self,
        env_idx: int
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Get episode statistics for a completed environment.

        Args:
            env_idx: Index of the environment

        Returns:
            Tuple of (total_reward, env_reward, shaped_bonus, actions_array, observations_array)
        """
        steps = self.step_counts[env_idx]
        env_reward = float(np.sum(self.rewards[env_idx, :steps]))
        shaped_bonus = float(self.shaped_bonuses[env_idx])
        total_reward = env_reward + shaped_bonus
        # Return slices of pre-allocated arrays (no copy needed for read-only use)
        actions_array = self.actions[env_idx, :steps].copy()
        observations_array = self.observations[env_idx, :steps].copy()

        return total_reward, env_reward, shaped_bonus, actions_array, observations_array
