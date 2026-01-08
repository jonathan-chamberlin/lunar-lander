"""Replay buffer for experience replay in TD3 training."""

import random
from typing import Optional

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

    def sample_raw(self, batch_size: int) -> list[Experience]:
        """Sample a random batch of experiences as a list.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of Experience tuples
        """
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_batch_size)

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


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized experience replay (PER).

    Note: This is a stub for future implementation.
    Currently behaves the same as ReplayBuffer.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ) -> None:
        super().__init__(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.priorities: list[float] = []
        self.frame = 0

    def push(self, experience: Experience, priority: Optional[float] = None) -> None:
        """Add an experience with priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        if priority is None:
            priority = max_priority

        if len(self.buffer) < self.capacity:
            self.priorities.append(priority)
        else:
            self.priorities[self.position] = priority

        super().push(experience)

    # TODO: Implement prioritized sampling based on TD-error
