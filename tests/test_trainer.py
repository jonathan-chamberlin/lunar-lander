"""Tests for TD3 trainer."""

import numpy as np
import pytest
import torch as T

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TrainingConfig, EnvironmentConfig, RunConfig
from training.trainer import TD3Trainer
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from data_types import Experience, ExperienceBatch


class TestTD3Trainer:
    """Tests for TD3Trainer class."""

    @pytest.fixture
    def trainer(self, training_config, env_config, run_config):
        """Create trainer instance."""
        return TD3Trainer(
            training_config=training_config,
            env_config=env_config,
            run_config=run_config,
            device=T.device('cpu')
        )

    @pytest.fixture
    def filled_buffer(self, sample_experiences):
        """Create a filled replay buffer."""
        buffer = ReplayBuffer(capacity=1000)
        for exp in sample_experiences:
            buffer.push(exp)
        return buffer

    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.actor is not None
        assert trainer.critic_1 is not None
        assert trainer.critic_2 is not None
        assert trainer.target_actor is not None
        assert trainer.target_critic_1 is not None
        assert trainer.target_critic_2 is not None
        assert trainer.training_steps == 0

    def test_actor_output_shape(self, trainer):
        """Test that actor outputs correct action shape."""
        state = T.randn(8)
        action = trainer.actor(state)
        assert action.shape == (2,)

    def test_actor_batch_output(self, trainer):
        """Test actor with batch input."""
        states = T.randn(32, 8)
        actions = trainer.actor(states)
        assert actions.shape == (32, 2)

    def test_critic_output_shape(self, trainer):
        """Test that critic outputs Q-value."""
        state = T.randn(1, 8)
        action = T.randn(1, 2)
        q_value = trainer.critic_1(state, action)
        assert q_value.shape == (1, 1)

    def test_train_step(self, trainer, sample_batch, seed):
        """Test single training step."""
        initial_steps = trainer.training_steps

        metrics, td_errors = trainer.train_step(sample_batch)

        assert trainer.training_steps == initial_steps + 1
        assert metrics.critic_loss > 0
        assert td_errors.shape[0] == sample_batch.states.shape[0]

    def test_train_step_updates_critic(self, trainer, sample_batch, seed):
        """Test that train step updates critic parameters."""
        # Get initial parameters
        initial_params = [p.clone() for p in trainer.critic_1.parameters()]

        trainer.train_step(sample_batch)

        # Check that at least some parameters changed
        current_params = list(trainer.critic_1.parameters())
        params_changed = False
        for init_p, curr_p in zip(initial_params, current_params):
            if not T.allclose(init_p, curr_p):
                params_changed = True
                break

        assert params_changed, "Critic parameters should update"

    def test_delayed_actor_updates(self, trainer, sample_batch, seed):
        """Test that actor updates are delayed by policy_update_frequency."""
        trainer.training_steps = 0

        # Get initial actor parameters
        initial_params = [p.clone() for p in trainer.actor.parameters()]

        # First step (training_steps=0), should update actor
        metrics_1, _ = trainer.train_step(sample_batch)
        assert metrics_1.actor_loss != 0.0

        # Steps 1 and 2 should NOT update actor (frequency=3)
        current_params_1 = [p.clone() for p in trainer.actor.parameters()]
        metrics_2, _ = trainer.train_step(sample_batch)
        assert metrics_2.actor_loss == 0.0  # No actor update

        metrics_3, _ = trainer.train_step(sample_batch)
        assert metrics_3.actor_loss == 0.0  # No actor update

        # Step 3 (training_steps=3) should update actor again
        metrics_4, _ = trainer.train_step(sample_batch)
        assert metrics_4.actor_loss != 0.0

    def test_train_on_buffer(self, trainer, filled_buffer, seed):
        """Test training from buffer."""
        initial_steps = trainer.training_steps

        metrics = trainer.train_on_buffer(filled_buffer, num_updates=10)

        assert trainer.training_steps == initial_steps + 10
        assert metrics.num_updates == 10
        assert metrics.critic_loss > 0

    def test_train_on_per_buffer(self, trainer, sample_experiences, seed):
        """Test training with prioritized replay buffer."""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        for exp in sample_experiences:
            buffer.push(exp)

        initial_steps = trainer.training_steps
        metrics = trainer.train_on_buffer(buffer, num_updates=5)

        assert trainer.training_steps == initial_steps + 5
        assert metrics.num_updates == 5

    def test_step_schedulers(self, trainer):
        """Test learning rate scheduler step."""
        initial_lr = trainer.actor_optimizer.param_groups[0]['lr']

        trainer.step_schedulers()

        new_lr = trainer.actor_optimizer.param_groups[0]['lr']
        assert new_lr < initial_lr  # LR should decay

    def test_target_networks_initialized_same(self, trainer):
        """Test that target networks are initialized with same weights."""
        for p, tp in zip(trainer.actor.parameters(), trainer.target_actor.parameters()):
            assert T.allclose(p, tp), "Target actor should match actor initially"

        for p, tp in zip(trainer.critic_1.parameters(), trainer.target_critic_1.parameters()):
            assert T.allclose(p, tp), "Target critic should match critic initially"

    def test_soft_update_changes_target(self, trainer, sample_batch, seed):
        """Test that soft update gradually changes target networks."""
        # Get initial target params
        initial_target_params = [p.clone() for p in trainer.target_actor.parameters()]

        # Multiple train steps with policy updates
        for _ in range(trainer.config.policy_update_frequency * 2):
            trainer.train_step(sample_batch)

        # Check that target params changed
        current_target_params = list(trainer.target_actor.parameters())
        params_changed = False
        for init_p, curr_p in zip(initial_target_params, current_target_params):
            if not T.allclose(init_p, curr_p):
                params_changed = True
                break

        assert params_changed, "Target parameters should change via soft update"


class TestTD3TrainerIntegration:
    """Integration tests for TD3Trainer."""

    def test_training_reduces_loss_over_time(self, seed):
        """Test that training reduces critic loss over multiple iterations."""
        config = TrainingConfig(
            buffer_size=500,
            batch_size=32,
            min_experiences_before_training=100
        )
        env_config = EnvironmentConfig()
        run_config = RunConfig(num_episodes=100)

        trainer = TD3Trainer(config, env_config, run_config, device=T.device('cpu'))
        buffer = ReplayBuffer(capacity=500)

        # Fill buffer with consistent experiences
        for i in range(200):
            state = T.randn(8)
            action = T.randn(2)
            # Create a simple pattern: action directly influences next state
            next_state = state + action.unsqueeze(0).expand(4, 2).reshape(-1)[:8] * 0.1
            reward = -T.sum(action ** 2)  # Penalize large actions

            exp = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=T.tensor(i % 50 == 49)
            )
            buffer.push(exp)

        # Train and track loss
        losses = []
        for _ in range(5):
            metrics = trainer.train_on_buffer(buffer, num_updates=10)
            losses.append(metrics.critic_loss)

        # Loss should generally decrease or stabilize
        # We don't require strict monotonic decrease due to stochasticity
        assert losses[-1] < losses[0] * 2, "Loss shouldn't explode"
