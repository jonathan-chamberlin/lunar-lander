"""Tests for environment utilities."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.environment import shape_reward, compute_noise_scale, EpisodeManager


class TestShapeReward:
    """Tests for reward shaping function."""

    def test_time_penalty_always_applied(self):
        """Test that time penalty is always applied."""
        state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        base_reward = 0.0

        shaped = shape_reward(state, base_reward, terminated=False)

        # Should include time penalty of -0.05
        assert shaped < base_reward

    def test_low_altitude_bonus_when_descending(self):
        """Test bonus for low altitude only when descending."""
        # Descending at low altitude
        descending_state = np.array([0.0, 0.2, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
        shaped_descending = shape_reward(descending_state, 0.0, terminated=False)

        # Hovering at low altitude (not descending)
        hovering_state = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        shaped_hovering = shape_reward(hovering_state, 0.0, terminated=False)

        # Descending should get more reward
        assert shaped_descending > shaped_hovering

    def test_leg_contact_bonus_when_descending(self):
        """Test leg contact bonuses only when descending."""
        # Single leg contact while descending
        one_leg_state = np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 1.0, 0.0])
        shaped_one_leg = shape_reward(one_leg_state, 0.0, terminated=False)

        # Both legs while descending
        both_legs_state = np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 1.0, 1.0])
        shaped_both_legs = shape_reward(both_legs_state, 0.0, terminated=False)

        # No legs while descending
        no_legs_state = np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
        shaped_no_legs = shape_reward(no_legs_state, 0.0, terminated=False)

        assert shaped_both_legs > shaped_one_leg > shaped_no_legs

    def test_stability_bonus_when_descending(self):
        """Test stability bonus for upright angle when descending."""
        # Upright while descending
        upright_state = np.array([0.0, 0.5, 0.0, -0.1, 0.05, 0.0, 0.0, 0.0])
        shaped_upright = shape_reward(upright_state, 0.0, terminated=False)

        # Tilted while descending
        tilted_state = np.array([0.0, 0.5, 0.0, -0.1, 0.5, 0.0, 0.0, 0.0])
        shaped_tilted = shape_reward(tilted_state, 0.0, terminated=False)

        assert shaped_upright > shaped_tilted

    def test_terminal_landing_bonus(self):
        """Test big bonus for successful terminal landing."""
        # Successful landing: terminated, both legs, positive base reward
        landing_state = np.array([0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 1.0, 1.0])

        shaped_landing = shape_reward(landing_state, base_reward=100.0, terminated=True)

        # Should include terminal bonus of +100
        assert shaped_landing > 150  # Base 100 + terminal bonus 100 - time penalty

    def test_no_terminal_bonus_on_crash(self):
        """Test no terminal bonus when crashing (negative base reward)."""
        # Crash: terminated, but negative base reward
        crash_state = np.array([0.0, 0.0, 0.0, -2.0, 0.8, 0.0, 0.0, 0.0])

        shaped_crash = shape_reward(crash_state, base_reward=-100.0, terminated=True)

        # Should not include terminal bonus
        assert shaped_crash < 0


class TestComputeNoiseScale:
    """Tests for noise scale computation."""

    def test_initial_scale(self):
        """Test that initial episode uses initial scale."""
        scale = compute_noise_scale(
            episode=0,
            initial_scale=1.0,
            final_scale=0.2,
            decay_episodes=300
        )
        assert scale == 1.0

    def test_final_scale(self):
        """Test that scale reaches final value after decay."""
        scale = compute_noise_scale(
            episode=300,
            initial_scale=1.0,
            final_scale=0.2,
            decay_episodes=300
        )
        assert scale == 0.2

    def test_scale_after_decay(self):
        """Test that scale stays at final after decay episodes."""
        scale = compute_noise_scale(
            episode=500,
            initial_scale=1.0,
            final_scale=0.2,
            decay_episodes=300
        )
        assert scale == 0.2

    def test_linear_decay(self):
        """Test that decay is linear."""
        # Halfway through decay
        scale = compute_noise_scale(
            episode=150,
            initial_scale=1.0,
            final_scale=0.2,
            decay_episodes=300
        )
        # Expected: 1.0 - (1.0 - 0.2) * 0.5 = 1.0 - 0.4 = 0.6
        assert abs(scale - 0.6) < 0.001


class TestEpisodeManager:
    """Tests for EpisodeManager class."""

    @pytest.fixture
    def manager(self):
        """Create episode manager."""
        return EpisodeManager(num_envs=4, max_episode_length=100)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.num_envs == 4
        assert manager.max_episode_length == 100
        assert np.all(manager.step_counts == 0)

    def test_add_step(self, manager):
        """Test adding a step to an environment."""
        action = np.array([0.5, -0.3])
        observation = np.zeros(8)

        manager.add_step(
            env_idx=0,
            reward=1.0,
            shaped_reward=1.5,
            action=action,
            observation=observation
        )

        assert manager.step_counts[0] == 1
        assert manager.shaped_bonuses[0] == 0.5  # shaped - original

    def test_reset_env(self, manager):
        """Test resetting a single environment."""
        action = np.array([0.5, -0.3])
        observation = np.zeros(8)

        # Add some steps
        for _ in range(10):
            manager.add_step(0, 1.0, 1.5, action, observation)

        assert manager.step_counts[0] == 10

        manager.reset_env(0)

        assert manager.step_counts[0] == 0
        assert manager.shaped_bonuses[0] == 0.0

    def test_get_episode_stats(self, manager):
        """Test getting episode statistics."""
        action = np.array([0.5, -0.3])
        observation = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        for i in range(5):
            manager.add_step(
                env_idx=0,
                reward=float(i),  # 0, 1, 2, 3, 4
                shaped_reward=float(i + 1),  # 1, 2, 3, 4, 5
                action=action,
                observation=observation * (i + 1)
            )

        total_reward, env_reward, shaped_bonus, actions, observations, duration = \
            manager.get_episode_stats(0)

        assert env_reward == 10.0  # 0+1+2+3+4
        assert shaped_bonus == 5.0  # (1-0)+(2-1)+(3-2)+(4-3)+(5-4)
        assert total_reward == 15.0  # env_reward + shaped_bonus
        assert actions.shape == (5, 2)
        assert observations.shape == (5, 8)
        assert duration >= 0

    def test_multiple_envs_independent(self, manager):
        """Test that multiple environments are tracked independently."""
        action = np.array([0.5, -0.3])
        observation = np.zeros(8)

        # Add different number of steps to each env
        for env_idx in range(4):
            for _ in range(env_idx + 1):  # 1, 2, 3, 4 steps
                manager.add_step(env_idx, 1.0, 1.0, action, observation)

        assert manager.step_counts[0] == 1
        assert manager.step_counts[1] == 2
        assert manager.step_counts[2] == 3
        assert manager.step_counts[3] == 4

    def test_max_episode_length_cap(self, manager):
        """Test that steps beyond max_episode_length are not recorded."""
        action = np.array([0.5, -0.3])
        observation = np.zeros(8)

        # Try to add more steps than max_episode_length
        for _ in range(150):
            manager.add_step(0, 1.0, 1.0, action, observation)

        # Should be capped at max_episode_length
        assert manager.step_counts[0] == 100  # max_episode_length
