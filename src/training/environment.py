"""Environment management and reward shaping for Lunar Lander."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Set

import gymnasium as gym
import numpy as np

from config import RunConfig, EnvironmentConfig, RewardShapingConfig

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
        env = gym.make(env_config.env_name, render_mode="rgb_array",
                       max_episode_steps=env_config.max_episode_steps)
        if run_config.framerate is not None:
            env.metadata["render_fps"] = run_config.framerate
        logger.info(f"Created environment with rgb_array rendering (max_steps={env_config.max_episode_steps})")
    else:
        env = gym.make(env_config.env_name, max_episode_steps=env_config.max_episode_steps)
        logger.info(f"Created environment without rendering (max_steps={env_config.max_episode_steps})")

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
    config: RewardShapingConfig,
    step: int = 0
) -> float:
    """Apply configurable reward shaping to provide intermediate learning signals.

    LunarLander state format:
    [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]

    Shaping rewards (configurable, gated on descending to prevent hovering):
    - Time penalty: -0.05 per step (discourages hovering/long episodes)
    - Altitude bonus: +0.5 when low (y_pos < 0.25) and descending
    - Leg contact bonus: +2/+5 for one/both legs when descending
    - Stability bonus: +0.3/+0.1 for upright when descending
    - Terminal landing bonus: +100 for successful landing (ALWAYS enabled)

    Args:
        state: Current state observation
        base_reward: Original reward from environment
        terminated: Whether episode has terminated (for landing bonus)
        config: RewardShapingConfig controlling which components are enabled
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

    # Component 1: Time penalty to discourage hovering (-0.05 per step)
    if config.time_penalty:
        shaped_reward -= 0.05

    # All per-step bonuses ONLY apply if descending (prevents hover exploitation)
    is_descending = y_vel < -0.05

    if is_descending:
        # Component 2: Altitude bonus for being close to ground
        if config.altitude_bonus and y_pos < 0.25:
            shaped_reward += 0.5

        # Component 3: Leg contact bonus
        if config.leg_contact:
            if leg1_contact and leg2_contact:
                shaped_reward += 5
            elif leg1_contact or leg2_contact:
                shaped_reward += 2

        # Component 4: Stability bonus for staying upright (angle near 0)
        if config.stability:
            if abs(angle) < 0.1:
                shaped_reward += 0.3
            elif abs(angle) < 0.2:
                shaped_reward += 0.1

    # Terminal landing bonus: big reward for successful landing (ALWAYS enabled)
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
