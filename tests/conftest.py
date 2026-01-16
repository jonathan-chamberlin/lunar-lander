"""Pytest fixtures for lunar lander tests."""

import numpy as np
import pytest
import torch as T

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, TrainingConfig, NoiseConfig, RunConfig, EnvironmentConfig
from data_types import Experience, ExperienceBatch


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    T.manual_seed(42)
    return 42


@pytest.fixture
def default_config():
    """Create default configuration for testing."""
    return Config()


@pytest.fixture
def training_config():
    """Create training configuration with small buffer for testing."""
    return TrainingConfig(
        buffer_size=1000,
        batch_size=32,
        min_experiences_before_training=100
    )


@pytest.fixture
def env_config():
    """Create environment configuration."""
    return EnvironmentConfig()


@pytest.fixture
def run_config():
    """Create run configuration with minimal episodes for testing."""
    return RunConfig(
        num_episodes=10,
        num_envs=2,
        render_mode='none',
        training_enabled=True
    )


@pytest.fixture
def sample_experience():
    """Create a sample experience tuple."""
    return Experience(
        state=T.randn(8),
        action=T.randn(2),
        reward=T.tensor(1.0),
        next_state=T.randn(8),
        done=T.tensor(False)
    )


@pytest.fixture
def sample_experiences(sample_experience):
    """Create a list of sample experiences."""
    experiences = []
    for i in range(100):
        exp = Experience(
            state=T.randn(8),
            action=T.randn(2),
            reward=T.tensor(float(i % 10 - 5)),  # Rewards from -5 to 4
            next_state=T.randn(8),
            done=T.tensor(i % 10 == 9)  # Done every 10th experience
        )
        experiences.append(exp)
    return experiences


@pytest.fixture
def sample_batch():
    """Create a sample experience batch."""
    batch_size = 32
    return ExperienceBatch(
        states=T.randn(batch_size, 8),
        actions=T.randn(batch_size, 2),
        rewards=T.randn(batch_size),
        next_states=T.randn(batch_size, 8),
        dones=T.zeros(batch_size, dtype=T.bool)
    )


@pytest.fixture
def sample_observations():
    """Create sample observation trajectory for behavior analysis."""
    # 100 timesteps of observations (x, y, vx, vy, angle, ang_vel, leg1, leg2)
    n_steps = 100
    observations = np.zeros((n_steps, 8), dtype=np.float32)

    # Simulate a controlled descent landing
    for i in range(n_steps):
        t = i / n_steps
        observations[i] = [
            0.1 * np.sin(t * 2),  # x: slight drift
            1.0 - t * 0.8,        # y: descent from 1.0 to 0.2
            0.05 * np.cos(t * 2), # vx: slight horizontal motion
            -0.3,                 # vy: constant descent velocity
            0.1 * np.sin(t * 3),  # angle: slight wobble
            0.1 * np.cos(t * 3),  # angular_vel
            1.0 if i > 80 else 0.0,  # leg1 contact
            1.0 if i > 82 else 0.0,  # leg2 contact
        ]

    return observations


@pytest.fixture
def sample_actions():
    """Create sample action trajectory."""
    n_steps = 100
    actions = np.zeros((n_steps, 2), dtype=np.float32)

    for i in range(n_steps):
        t = i / n_steps
        # Increasing main thrust as descent progresses
        actions[i, 0] = 0.3 + 0.4 * t
        # Small side corrections
        actions[i, 1] = 0.1 * np.sin(t * 5)

    return actions


@pytest.fixture
def crash_observations():
    """Create observations for a crash scenario."""
    n_steps = 50
    observations = np.zeros((n_steps, 8), dtype=np.float32)

    for i in range(n_steps):
        t = i / n_steps
        observations[i] = [
            0.0,                  # x: centered
            1.0 - t * 1.5,        # y: rapid descent
            0.0,                  # vx: no horizontal
            -1.5,                 # vy: fast descent (crash velocity)
            0.8 * t,              # angle: tilting
            0.5,                  # angular_vel: spinning
            0.0,                  # leg1: no contact
            0.0,                  # leg2: no contact
        ]

    return observations


@pytest.fixture
def crash_actions():
    """Create actions for a crash scenario."""
    n_steps = 50
    actions = np.zeros((n_steps, 2), dtype=np.float32)

    for i in range(n_steps):
        # No thrust - freefall
        actions[i, 0] = 0.0
        actions[i, 1] = 0.0

    return actions
