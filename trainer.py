"""TD3 (Twin Delayed DDPG) Trainer for continuous control."""

import logging
from typing import Optional

import torch as T
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from config import TrainingConfig, EnvironmentConfig, RunConfig
from network import ActorNetwork, CriticNetwork, soft_update, hard_update
from replay_buffer import ReplayBuffer
from data_types import ExperienceBatch, TrainingMetrics, AggregatedTrainingMetrics

logger = logging.getLogger(__name__)


class TD3Trainer:
    """Twin Delayed DDPG trainer for continuous control tasks.

    Implements TD3 algorithm with:
    - Clipped double Q-learning (two critics, take minimum)
    - Delayed policy updates (update actor less frequently than critics)
    - Target policy smoothing (add noise to target actions)
    - Learning rate scheduling (exponential decay)

    Args:
        training_config: Training hyperparameters
        env_config: Environment configuration (state/action dimensions)
        run_config: Run configuration (for LR scheduling based on num_episodes)
        device: Torch device to use (default: CPU)
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        env_config: EnvironmentConfig,
        run_config: Optional[RunConfig] = None,
        device: Optional[T.device] = None
    ) -> None:
        self.config = training_config
        self.env_config = env_config
        self.run_config = run_config or RunConfig()
        self.device = device or T.device('cpu')

        # Initialize networks
        self.actor = ActorNetwork(
            env_config.state_dim,
            env_config.action_dim
        ).to(self.device)

        self.critic_1 = CriticNetwork(
            env_config.state_dim,
            env_config.action_dim
        ).to(self.device)

        self.critic_2 = CriticNetwork(
            env_config.state_dim,
            env_config.action_dim
        ).to(self.device)

        # Initialize target networks
        self.target_actor = ActorNetwork(
            env_config.state_dim,
            env_config.action_dim
        ).to(self.device)

        self.target_critic_1 = CriticNetwork(
            env_config.state_dim,
            env_config.action_dim
        ).to(self.device)

        self.target_critic_2 = CriticNetwork(
            env_config.state_dim,
            env_config.action_dim
        ).to(self.device)

        # Copy weights to target networks
        hard_update(self.actor, self.target_actor)
        hard_update(self.critic_1, self.target_critic_1)
        hard_update(self.critic_2, self.target_critic_2)

        # Initialize optimizers with weight decay for regularization
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=training_config.actor_lr,
            weight_decay=1e-5
        )
        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(),
            lr=training_config.critic_lr,
            weight_decay=1e-5
        )
        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(),
            lr=training_config.critic_lr,
            weight_decay=1e-5
        )

        # Learning rate schedulers - decay to 20% of initial LR over training
        # gamma = 0.2^(1/num_episodes) gives final_lr = initial_lr * 0.2
        num_episodes = self.run_config.num_episodes
        lr_decay_gamma = 0.2 ** (1.0 / num_episodes) if num_episodes > 0 else 0.999
        self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=lr_decay_gamma)
        self.critic_1_scheduler = ExponentialLR(self.critic_1_optimizer, gamma=lr_decay_gamma)
        self.critic_2_scheduler = ExponentialLR(self.critic_2_optimizer, gamma=lr_decay_gamma)

        # Training state
        self.training_steps = 0

    def train_step(self, batch: ExperienceBatch) -> TrainingMetrics:
        """Perform one TD3 training update.

        Args:
            batch: Batch of experiences to train on

        Returns:
            TrainingMetrics with loss values and gradient norms
        """
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device)

        # === CRITIC TRAINING ===
        with T.no_grad():
            # Get next actions from target actor
            next_actions = self.target_actor(next_states)

            # Add clipped noise for target policy smoothing
            noise = T.randn_like(next_actions) * self.config.target_policy_noise
            noise = T.clamp(noise, -self.config.target_noise_clip, self.config.target_noise_clip)
            next_actions_noisy = next_actions + noise

            # Clamp actions to valid range
            next_actions_noisy[:, 0] = T.clamp(next_actions_noisy[:, 0], -1.0, 1.0)
            next_actions_noisy[:, 1] = T.clamp(next_actions_noisy[:, 1], -1.0, 1.0)

            # Compute target Q-values (take minimum of two critics)
            target_q1 = self.target_critic_1(next_states, next_actions_noisy)
            target_q2 = self.target_critic_2(next_states, next_actions_noisy)
            target_q = T.min(target_q1, target_q2)

            # Compute TD target
            done_mask = dones.unsqueeze(-1).float()
            target_q_value = rewards.unsqueeze(-1) + self.config.gamma * target_q * (1.0 - done_mask)

        # Compute current Q-values
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        # Compute critic losses using Smooth L1 (Huber) for robustness to outliers
        critic_loss_1 = F.smooth_l1_loss(current_q1, target_q_value)
        critic_loss_2 = F.smooth_l1_loss(current_q2, target_q_value)

        # Update critic 1
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        critic_grad_1 = T.nn.utils.clip_grad_norm_(
            self.critic_1.parameters(),
            self.config.gradient_clip_value
        )
        self.critic_1_optimizer.step()

        # Update critic 2
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        critic_grad_2 = T.nn.utils.clip_grad_norm_(
            self.critic_2.parameters(),
            self.config.gradient_clip_value
        )
        self.critic_2_optimizer.step()

        # Average critic metrics
        critic_loss = (critic_loss_1.item() + critic_loss_2.item()) / 2.0
        critic_grad_norm = (critic_grad_1.item() + critic_grad_2.item()) / 2.0

        # Initialize actor metrics
        actor_loss = 0.0
        actor_grad_norm = 0.0

        # === DELAYED ACTOR TRAINING ===
        if self.training_steps % self.config.policy_update_frequency == 0:
            # Compute actor loss (negative Q-value)
            predicted_actions = self.actor(states)
            actor_loss_tensor = -self.critic_1(states, predicted_actions).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss_tensor.backward()
            actor_grad_norm = T.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.config.gradient_clip_value
            ).item()
            self.actor_optimizer.step()

            actor_loss = actor_loss_tensor.item()

            # Soft update target networks
            soft_update(self.actor, self.target_actor, self.config.tau)
            soft_update(self.critic_1, self.target_critic_1, self.config.tau)
            soft_update(self.critic_2, self.target_critic_2, self.config.tau)

        self.training_steps += 1

        # Compute average Q-value for diagnostics
        avg_q = ((current_q1.mean() + current_q2.mean()) / 2.0).item()

        return TrainingMetrics(
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            avg_q_value=avg_q,
            critic_grad_norm=critic_grad_norm,
            actor_grad_norm=actor_grad_norm
        )

    def train_on_buffer(
        self,
        replay_buffer: ReplayBuffer,
        num_updates: int
    ) -> AggregatedTrainingMetrics:
        """Perform multiple training updates from the replay buffer.

        Args:
            replay_buffer: Buffer to sample experiences from
            num_updates: Number of gradient updates to perform

        Returns:
            AggregatedTrainingMetrics with averaged metrics
        """
        metrics_list = []

        for _ in range(num_updates):
            batch = replay_buffer.sample(self.config.batch_size)
            metrics = self.train_step(batch)
            metrics_list.append(metrics)

        return AggregatedTrainingMetrics.from_metrics_list(metrics_list)

    def step_schedulers(self) -> None:
        """Step learning rate schedulers. Call once per episode."""
        self.actor_scheduler.step()
        self.critic_1_scheduler.step()
        self.critic_2_scheduler.step()

    def save(self, path: str) -> None:
        """Save model weights to file.

        Args:
            path: Path to save the model (without extension)
        """
        T.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_1_scheduler': self.critic_1_scheduler.state_dict(),
            'critic_2_scheduler': self.critic_2_scheduler.state_dict(),
            'training_steps': self.training_steps
        }, f"{path}.pt")
        logger.info(f"Model saved to {path}.pt")

    def load(self, path: str) -> None:
        """Load model weights from file.

        Args:
            path: Path to load the model from (without extension)
        """
        checkpoint = T.load(f"{path}.pt", map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        # Load scheduler state if available (backwards compatible)
        if 'actor_scheduler' in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
            self.critic_1_scheduler.load_state_dict(checkpoint['critic_1_scheduler'])
            self.critic_2_scheduler.load_state_dict(checkpoint['critic_2_scheduler'])
        self.training_steps = checkpoint['training_steps']

        logger.info(f"Model loaded from {path}.pt (step {self.training_steps})")
