"""Training components for TD3 reinforcement learning."""

from training.network import ActorNetwork, CriticNetwork, OUActionNoise, soft_update, hard_update
from training.trainer import TD3Trainer
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, SumTree
from training.environment import (
    EnvironmentBundle,
    create_environments,
    shape_reward,
    compute_noise_scale,
    EpisodeManager
)

__all__ = [
    'ActorNetwork',
    'CriticNetwork',
    'OUActionNoise',
    'soft_update',
    'hard_update',
    'TD3Trainer',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'SumTree',
    'EnvironmentBundle',
    'create_environments',
    'shape_reward',
    'compute_noise_scale',
    'EpisodeManager',
]
