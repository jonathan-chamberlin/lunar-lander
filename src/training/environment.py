"""Environment management and reward shaping for Lunar Lander."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Set

import gymnasium as gym
import numpy as np

from config import RunConfig, EnvironmentConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentBundle:
    """Bundle containing the training environment.

    Attributes:
        env: Single Gymnasium environment for training
        render_episodes: Set of episode numbers to render
        should_render: Whether rendering is enabled for this bundle
    """

    env: gym.Env
    render_episodes: Set[int]
    should_render: bool


@contextmanager
def create_environments(
    run_config: RunConfig,
    env_config: EnvironmentConfig
) -> Generator[EnvironmentBundle, None, None]:
    """Context manager for creating and managing Gymnasium environment.

    Creates a single environment for training. If render_mode is 'all' or 'custom',
    the environment is created with rgb_array rendering. If 'none', no rendering.

    Args:
        run_config: Configuration for training runs
        env_config: Environment configuration

    Yields:
        EnvironmentBundle containing the training environment
    """
    should_render = run_config.render_mode != 'none'

    # Create environment with appropriate render mode
    if should_render:
        env = gym.make(env_config.env_name, render_mode="rgb_array")
        if run_config.framerate is not None:
            env.metadata["render_fps"] = run_config.framerate
        logger.info("Created environment with rgb_array rendering")
    else:
        env = gym.make(env_config.env_name)
        logger.info("Created environment without rendering (headless mode)")

    bundle = EnvironmentBundle(
        env=env,
        render_episodes=set(run_config.render_episodes),
        should_render=should_render
    )

    try:
        yield bundle
    finally:
        env.close()
        logger.info("Environment closed")


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
