"""Neural network architectures for TD3 actor-critic reinforcement learning."""

from typing import Optional

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from config import NoiseConfig


class ActorNetwork(nn.Module):
    """Actor network that maps states to continuous actions.

    Architecture: state -> 400 -> 300 -> action
    Uses separate output heads for main and side thrusters with tanh activation.

    Args:
        state_dim: Dimension of the state/observation space
        action_dim: Dimension of the action space
        hidden1: Size of first hidden layer (default: 400)
        hidden2: Size of second hidden layer (default: 300)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 400,
        hidden2: int = 300
    ) -> None:
        super().__init__()

        self.layer1 = nn.Linear(state_dim, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)

        # Separate output layers for each action dimension
        self.main_engine_layer = nn.Linear(hidden2, 1)  # Main engine [-1, 1]
        self.side_engine_layer = nn.Linear(hidden2, 1)  # Side engine [-1, 1]

        # Initialize biases to 0 so tanh(0) = 0 (no thrust initially)
        nn.init.constant_(self.main_engine_layer.bias, 0.0)
        nn.init.constant_(self.side_engine_layer.bias, 0.0)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Forward pass to compute actions from states.

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Action tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        # Apply tanh activation for bounded actions
        main_engine = T.tanh(self.main_engine_layer(x))
        side_engine = T.tanh(self.side_engine_layer(x))

        # Concatenate to form action vector
        action = T.cat([main_engine, side_engine], dim=-1)
        return action


class CriticNetwork(nn.Module):
    """Critic network that estimates Q-values for state-action pairs.

    Architecture: state -> 400 -> concat(state_features, action) -> 300 -> Q-value

    Args:
        state_dim: Dimension of the state/observation space
        action_dim: Dimension of the action space
        hidden1: Size of first hidden layer (default: 400)
        hidden2: Size of second hidden layer (default: 300)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 400,
        hidden2: int = 300
    ) -> None:
        super().__init__()

        # State processing stream
        self.state_layer = nn.Linear(state_dim, hidden1)
        # Combined processing (state features + action)
        self.combined_layer = nn.Linear(hidden1 + action_dim, hidden2)
        # Output single Q-value
        self.output_layer = nn.Linear(hidden2, 1)

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        """Forward pass to compute Q-value from state-action pair.

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)

        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        state_features = F.relu(self.state_layer(state))
        combined = T.cat([state_features, action], dim=-1)
        x = F.relu(self.combined_layer(combined))
        q_value = self.output_layer(x)
        return q_value


class OUActionNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Generates noise that is correlated in time, which is beneficial for
    continuous control tasks with inertia.

    Args:
        config: NoiseConfig containing mu, sigma, theta, dt, x0, action_dimensions
        num_envs: Number of parallel environments to generate noise for
    """

    def __init__(self, config: NoiseConfig, num_envs: int = 1) -> None:
        self.sigma = config.sigma
        self.theta = config.theta
        self.dt = config.dt
        self.x0 = config.x0
        self.action_dimensions = config.action_dimensions
        self.num_envs = num_envs

        # Convert mu to numpy array
        if isinstance(config.mu, (int, float)):
            self.mu = np.array([config.mu] * config.action_dimensions)
        else:
            self.mu = np.array(config.mu)

        # Initialize noise state
        self.noise: np.ndarray = np.zeros((num_envs, config.action_dimensions))
        self.reset()

    def reset(self, env_idx: Optional[int] = None) -> None:
        """Reset noise state to initial value.

        Args:
            env_idx: If provided, reset only that environment's noise.
                     If None, reset all environments.
        """
        if env_idx is not None:
            if self.x0 is None:
                self.noise[env_idx] = self.mu.copy()
            else:
                self.noise[env_idx] = np.ones(self.action_dimensions) * self.x0
        else:
            if self.x0 is None:
                self.noise = np.tile(self.mu, (self.num_envs, 1))
            else:
                self.noise = np.ones((self.num_envs, self.action_dimensions)) * self.x0

    def generate(self) -> T.Tensor:
        """Generate noise for all environments.

        Returns:
            Noise tensor of shape (num_envs, action_dimensions)
        """
        random_noise = np.random.normal(size=(self.num_envs, self.action_dimensions))
        self.noise = (
            self.noise
            + self.theta * (self.mu - self.noise) * self.dt
            + self.sigma * np.sqrt(self.dt) * random_noise
        )
        return T.from_numpy(self.noise).float()

    def generate_single(self, env_idx: int = 0) -> T.Tensor:
        """Generate noise for a single environment.

        Args:
            env_idx: Index of the environment (default: 0)

        Returns:
            Noise tensor of shape (action_dimensions,)
        """
        random_noise = np.random.normal(size=(1, self.action_dimensions))
        single_noise = (
            self.noise[env_idx:env_idx + 1]
            + self.theta * (self.mu - self.noise[env_idx:env_idx + 1]) * self.dt
            + self.sigma * np.sqrt(self.dt) * random_noise
        )
        return T.from_numpy(single_noise[0]).float()

    # Aliases for backward compatibility
    def generate_noise(self) -> T.Tensor:
        """Alias for generate() - maintained for backward compatibility."""
        return self.generate()

    def generate_noise_single(self) -> T.Tensor:
        """Alias for generate_single() - maintained for backward compatibility."""
        return self.generate_single()


def soft_update(
    source: nn.Module,
    target: nn.Module,
    tau: float
) -> None:
    """Soft update target network parameters using Polyak averaging.

    target = tau * source + (1 - tau) * target

    Args:
        source: Source network to copy from
        target: Target network to update
        tau: Interpolation parameter (0 < tau <= 1)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Hard update target network parameters (direct copy).

    Args:
        source: Source network to copy from
        target: Target network to update
    """
    target.load_state_dict(source.state_dict())
