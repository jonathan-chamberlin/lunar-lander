"""Tests for configuration dataclasses."""

import pytest
from dataclasses import FrozenInstanceError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    Config, TrainingConfig, NoiseConfig, RunConfig,
    EnvironmentConfig, DisplayConfig
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TrainingConfig()
        assert config.actor_lr == 0.001
        assert config.critic_lr == 0.002
        assert config.gamma == 0.99
        assert config.tau == 0.005
        assert config.batch_size == 128
        assert config.buffer_size == 1 << 14  # 16384
        assert config.use_per is True

    def test_frozen_immutability(self):
        """Test that frozen dataclass cannot be modified."""
        config = TrainingConfig()
        with pytest.raises(FrozenInstanceError):
            config.actor_lr = 0.01

    def test_custom_values(self):
        """Test that custom values are accepted."""
        config = TrainingConfig(
            actor_lr=0.0005,
            critic_lr=0.001,
            batch_size=64
        )
        assert config.actor_lr == 0.0005
        assert config.critic_lr == 0.001
        assert config.batch_size == 64


class TestNoiseConfig:
    """Tests for NoiseConfig dataclass."""

    def test_default_values(self):
        """Test default noise parameters."""
        config = NoiseConfig()
        assert config.sigma == 0.3
        assert config.theta == 0.2
        assert config.noise_scale_initial == 1.0
        assert config.noise_scale_final == 0.2

    def test_mu_default(self):
        """Test that mu is a tuple of zeros."""
        config = NoiseConfig()
        assert config.mu == (0.0, 0.0)


class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_default_values(self):
        """Test default run configuration."""
        config = RunConfig()
        assert config.num_episodes == 5000
        assert config.num_envs == 8
        assert config.render_mode == 'all'
        assert config.training_enabled is True

    def test_render_mode_none(self):
        """Test render_mode='none' sets empty render_episodes."""
        config = RunConfig(render_mode='none', num_episodes=100)
        assert config.render_episodes == ()

    def test_render_mode_all(self):
        """Test render_mode='all' renders all episodes."""
        config = RunConfig(render_mode='all', num_episodes=100)
        assert len(config.render_episodes) == 100
        assert 0 in config.render_episodes
        assert 99 in config.render_episodes

    def test_render_mode_custom(self):
        """Test render_mode='custom' uses custom episodes."""
        custom_episodes = (0, 10, 50, 99)
        config = RunConfig(
            render_mode='custom',
            render_episodes_custom=custom_episodes,
            num_episodes=100
        )
        assert config.render_episodes == custom_episodes

    def test_invalid_render_mode_raises(self):
        """Test that invalid render_mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid render_mode"):
            RunConfig(render_mode='invalid')


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_default_values(self):
        """Test default environment configuration."""
        config = EnvironmentConfig()
        assert config.env_name == "LunarLanderContinuous-v3"
        assert config.state_dim == 8
        assert config.action_dim == 2
        assert config.success_threshold == 200.0


class TestDisplayConfig:
    """Tests for DisplayConfig dataclass."""

    def test_default_values(self):
        """Test default display configuration."""
        config = DisplayConfig()
        assert config.show_run_overlay is False
        assert config.font_size == 30
        assert config.font_color == (255, 255, 0)


class TestConfig:
    """Tests for master Config class."""

    def test_default_initialization(self):
        """Test that Config creates all sub-configs with defaults."""
        config = Config()
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.noise, NoiseConfig)
        assert isinstance(config.run, RunConfig)
        assert isinstance(config.environment, EnvironmentConfig)
        assert isinstance(config.display, DisplayConfig)

    def test_custom_sub_configs(self):
        """Test Config with custom sub-configurations."""
        training = TrainingConfig(actor_lr=0.0001)
        noise = NoiseConfig(sigma=0.5)

        config = Config(training=training, noise=noise)
        assert config.training.actor_lr == 0.0001
        assert config.noise.sigma == 0.5
        # Other configs should have defaults
        assert config.run.num_episodes == 5000
