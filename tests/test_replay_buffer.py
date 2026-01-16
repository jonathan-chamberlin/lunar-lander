"""Tests for replay buffer implementations."""

import numpy as np
import pytest
import torch as T

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, SumTree
from data_types import Experience


class TestReplayBuffer:
    """Tests for uniform ReplayBuffer."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=1000)
        assert len(buffer) == 0
        assert buffer.capacity == 1000

    def test_push_single(self, sample_experience):
        """Test pushing a single experience."""
        buffer = ReplayBuffer(capacity=100)
        buffer.push(sample_experience)
        assert len(buffer) == 1

    def test_push_multiple(self, sample_experiences):
        """Test pushing multiple experiences."""
        buffer = ReplayBuffer(capacity=100)
        for exp in sample_experiences:
            buffer.push(exp)
        assert len(buffer) == 100  # Capped at capacity

    def test_circular_buffer_overwrites(self):
        """Test that buffer overwrites oldest experiences when full."""
        buffer = ReplayBuffer(capacity=10)

        # Add 15 experiences
        for i in range(15):
            exp = Experience(
                state=T.tensor([float(i)]),
                action=T.tensor([0.0]),
                reward=T.tensor(float(i)),
                next_state=T.tensor([float(i + 1)]),
                done=T.tensor(False)
            )
            buffer.push(exp)

        # Buffer should contain only last 10
        assert len(buffer) == 10

        # Verify oldest (0-4) are gone, newest (5-14) remain
        rewards_in_buffer = [exp.reward.item() for exp in buffer.buffer]
        assert 0.0 not in rewards_in_buffer
        assert 14.0 in rewards_in_buffer

    def test_sample_returns_correct_shape(self, sample_experiences, seed):
        """Test that sample returns correctly shaped batch."""
        buffer = ReplayBuffer(capacity=100)
        for exp in sample_experiences:
            buffer.push(exp)

        batch = buffer.sample(32)
        assert batch.states.shape == (32, 8)
        assert batch.actions.shape == (32, 2)
        assert batch.rewards.shape == (32,)
        assert batch.next_states.shape == (32, 8)
        assert batch.dones.shape == (32,)

    def test_sample_smaller_than_buffer(self, sample_experiences, seed):
        """Test sampling when batch_size > buffer size."""
        buffer = ReplayBuffer(capacity=100)
        for exp in sample_experiences[:5]:  # Only 5 experiences
            buffer.push(exp)

        batch = buffer.sample(32)  # Request 32
        assert batch.states.shape[0] == 5  # Should return 5

    def test_is_ready(self, sample_experiences):
        """Test is_ready method."""
        buffer = ReplayBuffer(capacity=100)

        assert not buffer.is_ready(10)

        for exp in sample_experiences[:5]:
            buffer.push(exp)
        assert not buffer.is_ready(10)

        for exp in sample_experiences[5:15]:
            buffer.push(exp)
        assert buffer.is_ready(10)

    def test_clear(self, sample_experiences):
        """Test buffer clear."""
        buffer = ReplayBuffer(capacity=100)
        for exp in sample_experiences:
            buffer.push(exp)
        assert len(buffer) == 100

        buffer.clear()
        assert len(buffer) == 0


class TestSumTree:
    """Tests for SumTree data structure."""

    def test_initialization(self):
        """Test SumTree initialization."""
        tree = SumTree(capacity=10)
        assert tree.capacity == 10
        assert tree.total == 0.0

    def test_update(self):
        """Test priority update."""
        tree = SumTree(capacity=10)
        tree.update(0, 1.0)
        tree.update(1, 2.0)
        tree.update(2, 3.0)

        assert tree.total == 6.0
        assert tree[0] == 1.0
        assert tree[1] == 2.0
        assert tree[2] == 3.0

    def test_get_leaf_distribution(self, seed):
        """Test that get_leaf samples proportionally to priorities."""
        tree = SumTree(capacity=4)
        tree.update(0, 1.0)
        tree.update(1, 2.0)
        tree.update(2, 3.0)
        tree.update(3, 4.0)

        # Sample many times and check distribution
        counts = [0, 0, 0, 0]
        n_samples = 10000
        for _ in range(n_samples):
            value = np.random.uniform(0, tree.total)
            idx = tree.get_leaf(value)
            counts[idx] += 1

        # Expected proportions: 0.1, 0.2, 0.3, 0.4
        proportions = [c / n_samples for c in counts]

        # Check that proportions are approximately correct (within 5%)
        assert abs(proportions[0] - 0.1) < 0.05
        assert abs(proportions[1] - 0.2) < 0.05
        assert abs(proportions[2] - 0.3) < 0.05
        assert abs(proportions[3] - 0.4) < 0.05


class TestPrioritizedReplayBuffer:
    """Tests for PrioritizedReplayBuffer."""

    def test_initialization(self):
        """Test PER buffer initialization."""
        buffer = PrioritizedReplayBuffer(
            capacity=1000,
            alpha=0.6,
            beta_start=0.4
        )
        assert len(buffer) == 0
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4

    def test_push(self, sample_experience):
        """Test pushing experience with max priority."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        buffer.push(sample_experience)
        assert len(buffer) == 1
        assert buffer.max_priority == 1.0

    def test_sample_returns_weights_and_indices(self, sample_experiences, seed):
        """Test that PER sample returns weights and indices."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        for exp in sample_experiences:
            buffer.push(exp)

        batch = buffer.sample(32)
        assert batch.weights is not None
        assert batch.indices is not None
        assert batch.weights.shape == (32,)
        assert batch.indices.shape == (32,)

    def test_update_priorities(self, sample_experiences, seed):
        """Test priority update."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
        for exp in sample_experiences:
            buffer.push(exp)

        batch = buffer.sample(32)
        old_priorities = [buffer.tree[i] for i in batch.indices]

        # Update with new priorities
        new_priorities = np.ones(32) * 10.0
        buffer.update_priorities(batch.indices, new_priorities)

        updated_priorities = [buffer.tree[i] for i in batch.indices]
        assert all(p > old_p for p, old_p in zip(updated_priorities, old_priorities))

    def test_anneal_beta(self):
        """Test beta annealing."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            beta_start=0.4,
            beta_end=1.0
        )
        assert buffer.beta == 0.4

        buffer.anneal_beta(0.5)  # 50% progress
        assert buffer.beta == 0.7  # Halfway between 0.4 and 1.0

        buffer.anneal_beta(1.0)  # 100% progress
        assert buffer.beta == 1.0

    def test_is_ready(self, sample_experiences):
        """Test is_ready method."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        assert not buffer.is_ready(10)

        for exp in sample_experiences[:15]:
            buffer.push(exp)
        assert buffer.is_ready(10)

    def test_clear(self, sample_experiences):
        """Test buffer clear."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        for exp in sample_experiences:
            buffer.push(exp)

        buffer.clear()
        assert len(buffer) == 0
        assert buffer.max_priority == 1.0
