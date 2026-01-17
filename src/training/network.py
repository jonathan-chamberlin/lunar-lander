"""Neural network architectures for TD3 actor-critic reinforcement learning."""

from typing import Optional

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Actor network that maps states to continuous actions.

    Architecture: state -> 256 -> LayerNorm -> 128 -> LayerNorm -> action
    Uses separate output heads for main and side thrusters with tanh activation.
    Reduced network size for faster training (8D state space doesn't need large networks).

    Args:
        state_dim: Dimension of the state/observation space
        action_dim: Dimension of the action space
        hidden1: Size of first hidden layer (default: 256, reduced from 400)
        hidden2: Size of second hidden layer (default: 128, reduced from 300)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 256,
        hidden2: int = 128
    ) -> None:
        super().__init__()

        self.layer1 = nn.Linear(state_dim, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)

        # Separate output layers for each action dimension
        self.main_engine_layer = nn.Linear(hidden2, 1)  # Main engine [-1, 1]
        self.side_engine_layer = nn.Linear(hidden2, 1)  # Side engine [-1, 1]

        # Initialize weights with Xavier uniform for stable gradients
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.main_engine_layer.weight)
        nn.init.xavier_uniform_(self.side_engine_layer.weight)

        # Initialize all biases to zero (no action bias - let agent learn from scratch)
        nn.init.constant_(self.layer1.bias, 0.0)
        nn.init.constant_(self.layer2.bias, 0.0)
        nn.init.constant_(self.main_engine_layer.bias, 0.0)
        nn.init.constant_(self.side_engine_layer.bias, 0.0)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Forward pass to compute actions from states.

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Action tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        x = F.relu(self.ln1(self.layer1(state)))
        x = F.relu(self.ln2(self.layer2(x)))

        # Apply tanh activation for bounded actions
        main_engine = T.tanh(self.main_engine_layer(x))
        side_engine = T.tanh(self.side_engine_layer(x))

        # Concatenate to form action vector
        action = T.cat([main_engine, side_engine], dim=-1)
        return action


class CriticNetwork(nn.Module):
    """Critic network that estimates Q-values for state-action pairs.

    Architecture: state -> 256 -> LayerNorm -> concat(state_features, action) -> 128 -> LayerNorm -> Q-value
    Reduced network size for faster training.

    Args:
        state_dim: Dimension of the state/observation space
        action_dim: Dimension of the action space
        hidden1: Size of first hidden layer (default: 256, reduced from 400)
        hidden2: Size of second hidden layer (default: 128, reduced from 300)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden1: int = 256,
        hidden2: int = 128
    ) -> None:
        super().__init__()

        # State processing stream
        self.state_layer = nn.Linear(state_dim, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        # Combined processing (state features + action)
        self.combined_layer = nn.Linear(hidden1 + action_dim, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        # Output single Q-value
        self.output_layer = nn.Linear(hidden2, 1)

        # Initialize weights with Xavier uniform for stable gradients
        nn.init.xavier_uniform_(self.state_layer.weight)
        nn.init.xavier_uniform_(self.combined_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

        # Initialize biases to zero
        nn.init.constant_(self.state_layer.bias, 0.0)
        nn.init.constant_(self.combined_layer.bias, 0.0)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        """Forward pass to compute Q-value from state-action pair.

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)

        Returns:
            Q-value tensor of shape (batch_size, 1)
        """
        state_features = F.relu(self.ln1(self.state_layer(state)))
        combined = T.cat([state_features, action], dim=-1)
        x = F.relu(self.ln2(self.combined_layer(combined)))
        q_value = self.output_layer(x)
        return q_value


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
    # Use no_grad and in-place operations for speed
    with T.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Hard update target network parameters (direct copy).

    Args:
        source: Source network to copy from
        target: Target network to update
    """
    target.load_state_dict(source.state_dict())
