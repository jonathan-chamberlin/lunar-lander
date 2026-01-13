"""Replay buffer for experience replay in TD3 training."""

import random
from typing import Optional

import numpy as np
import torch as T

from data_types import Experience, ExperienceBatch


class ReplayBuffer:
    """Circular buffer for storing and sampling experiences.

    Uses a list with manual index management for O(1) random access.
    (deque + random.sample is O(n) which degrades performance)

    Args:
        capacity: Maximum number of experiences to store
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[Experience] = []
        self.position = 0

    def push(self, experience: Experience) -> None:
        """Add an experience to the buffer.

        Args:
            experience: The experience tuple to store
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> ExperienceBatch:
        """Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            ExperienceBatch with stacked tensors
        """
        actual_batch_size = min(batch_size, len(self.buffer))
        # Use numpy for faster random index generation
        indices = np.random.choice(len(self.buffer), actual_batch_size, replace=False)

        # Direct list comprehension is faster than zip(*samples)
        states = T.stack([self.buffer[i].state for i in indices])
        actions = T.stack([self.buffer[i].action for i in indices])
        rewards = T.stack([self.buffer[i].reward for i in indices])
        next_states = T.stack([self.buffer[i].next_state for i in indices])
        dones = T.stack([self.buffer[i].done for i in indices])

        return ExperienceBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """No-op for uniform replay buffer (for API compatibility)."""
        pass

    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training.

        Args:
            min_size: Minimum number of experiences required

        Returns:
            True if buffer has at least min_size experiences
        """
        return len(self.buffer) >= min_size

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
        self.position = 0


class SumTree:
    """Binary tree where parent nodes store sum of children.

    Enables O(log n) priority-based sampling with O(log n) updates.
    Leaf nodes store priorities, internal nodes store sums.

    Args:
        capacity: Maximum number of leaf nodes (experiences)
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Tree size: 2 * capacity - 1 (full binary tree with capacity leaves)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree (iterative for speed)."""
        while idx != 0:
            parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    def update(self, idx: int, priority: float) -> None:
        """Update priority at leaf index."""
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get_leaf(self, value: float) -> int:
        """Find leaf index for a given cumulative sum value.

        Args:
            value: Random value in [0, total_priority]

        Returns:
            Leaf index (data index, not tree index)
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                # Reached leaf level
                return parent - (self.capacity - 1)
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right

    @property
    def total(self) -> float:
        """Total priority (root node)."""
        return self.tree[0]

    def __getitem__(self, idx: int) -> float:
        """Get priority at leaf index."""
        return self.tree[idx + self.capacity - 1]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using Sum Tree.

    Samples experiences based on TD-error magnitude, allowing the agent
    to learn more from surprising/important transitions.

    Args:
        capacity: Maximum number of experiences to store
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta_start: Initial importance sampling weight
        beta_end: Final importance sampling weight
        epsilon: Small constant to prevent zero priorities
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-6
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.data: list[Optional[Experience]] = [None] * capacity
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, experience: Experience) -> None:
        """Add an experience with max priority (for exploration)."""
        # New experiences get max priority to ensure they're sampled
        priority = self.max_priority ** self.alpha
        self.tree.update(self.position, priority)
        self.data[self.position] = experience

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ExperienceBatch:
        """Sample a prioritized batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            ExperienceBatch with stacked tensors, weights, and indices
        """
        actual_batch_size = min(batch_size, self.size)
        indices = np.zeros(actual_batch_size, dtype=np.int32)
        priorities = np.zeros(actual_batch_size, dtype=np.float32)

        # Segment the total priority into batch_size ranges
        segment = self.tree.total / actual_batch_size

        # Generate all random values at once (vectorized)
        segment_starts = np.arange(actual_batch_size) * segment
        random_offsets = np.random.uniform(0, segment, size=actual_batch_size)
        values = segment_starts + random_offsets

        # Get leaf indices for all values
        for i, value in enumerate(values):
            idx = self.tree.get_leaf(value)
            idx = min(idx, self.size - 1)
            indices[i] = idx
            priorities[i] = self.tree[idx]

        # Compute importance sampling weights (vectorized)
        probabilities = priorities / (self.tree.total + 1e-8)
        weights = (self.size * probabilities) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)
        weights = T.from_numpy(weights).float()

        # Gather samples and stack tensors
        samples = [self.data[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        return ExperienceBatch(
            states=T.stack(states),
            actions=T.stack(actions),
            rewards=T.stack(rewards),
            next_states=T.stack(next_states),
            dones=T.stack(dones),
            weights=weights,
            indices=indices
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values (typically |TD-error| + epsilon)
        """
        # Vectorized clip operation
        priorities = np.clip(priorities, self.epsilon, 100.0)
        self.max_priority = max(self.max_priority, float(np.max(priorities)))

        # Update tree for each index (tree structure requires sequential updates)
        for idx, priority in zip(indices, priorities):
            self.tree.update(int(idx), priority ** self.alpha)

    def anneal_beta(self, progress: float) -> None:
        """Anneal beta towards 1.0 over training.

        Args:
            progress: Training progress in [0, 1]
        """
        self.beta = self.beta_start + progress * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return self.size >= min_size

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.tree = SumTree(self.capacity)
        self.data = [None] * self.capacity
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
