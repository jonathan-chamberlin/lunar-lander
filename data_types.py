"""Type definitions for the Lunar Lander TD3 training."""

from dataclasses import dataclass
from typing import NamedTuple

import torch as T


class Experience(NamedTuple):
    """A single transition in the replay buffer."""

    state: T.Tensor
    action: T.Tensor
    reward: T.Tensor
    next_state: T.Tensor
    done: T.Tensor


class ExperienceBatch(NamedTuple):
    """A batch of experiences for training."""

    states: T.Tensor
    actions: T.Tensor
    rewards: T.Tensor
    next_states: T.Tensor
    dones: T.Tensor


@dataclass
class TrainingMetrics:
    """Metrics from a single training step."""

    critic_loss: float
    actor_loss: float
    avg_q_value: float
    critic_grad_norm: float
    actor_grad_norm: float

    def __str__(self) -> str:
        return (
            f"Critic Loss: {self.critic_loss:.4f}, "
            f"Actor Loss: {self.actor_loss:.4f}, "
            f"Avg Q: {self.avg_q_value:.3f}"
        )


@dataclass
class AggregatedTrainingMetrics:
    """Aggregated metrics over multiple training steps."""

    critic_loss: float
    actor_loss: float
    avg_q_value: float
    critic_grad_norm: float
    actor_grad_norm: float
    num_updates: int
    num_actor_updates: int

    @classmethod
    def from_metrics_list(
        cls,
        metrics_list: list[TrainingMetrics]
    ) -> 'AggregatedTrainingMetrics':
        """Aggregate a list of training metrics."""
        if not metrics_list:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

        num_updates = len(metrics_list)
        actor_updates = [m for m in metrics_list if m.actor_loss != 0.0]
        num_actor_updates = len(actor_updates)

        return cls(
            critic_loss=sum(m.critic_loss for m in metrics_list) / num_updates,
            actor_loss=(
                sum(m.actor_loss for m in actor_updates) / num_actor_updates
                if num_actor_updates > 0 else 0.0
            ),
            avg_q_value=sum(m.avg_q_value for m in metrics_list) / num_updates,
            critic_grad_norm=sum(m.critic_grad_norm for m in metrics_list) / num_updates,
            actor_grad_norm=(
                sum(m.actor_grad_norm for m in actor_updates) / num_actor_updates
                if num_actor_updates > 0 else 0.0
            ),
            num_updates=num_updates,
            num_actor_updates=num_actor_updates
        )


@dataclass
class EpisodeResult:
    """Result of a completed episode."""

    episode_num: int
    total_reward: float
    env_reward: float
    shaped_bonus: float
    steps: int
    success: bool

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        return (
            f"Episode {self.episode_num}: {status}, "
            f"Reward: {self.total_reward:.1f} "
            f"(env: {self.env_reward:.1f}, shaped: {self.shaped_bonus:.1f})"
        )


@dataclass
class ActionStatistics:
    """Statistics about actions taken during an episode."""

    mean_magnitude: float
    std: float
    mean_main_thruster: float
    mean_side_thruster: float

    @classmethod
    def from_actions(cls, actions_array) -> 'ActionStatistics':
        """Compute statistics from an array of actions."""
        import numpy as np
        return cls(
            mean_magnitude=float(np.mean(np.abs(actions_array))),
            std=float(np.std(actions_array)),
            mean_main_thruster=float(np.mean(actions_array[:, 0])),
            mean_side_thruster=float(np.mean(actions_array[:, 1]))
        )
