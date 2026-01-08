"""Replay buffer for experience replay in TD3 training."""

import random

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
        samples = random.sample(self.buffer, actual_batch_size)

        states, actions, rewards, next_states, dones = zip(*samples)

        return ExperienceBatch(
            states=T.stack(states),
            actions=T.stack(actions),
            rewards=T.stack(rewards),
            next_states=T.stack(next_states),
            dones=T.stack(dones)
        )

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
