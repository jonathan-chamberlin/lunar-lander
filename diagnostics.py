"""Diagnostics tracking and reporting for TD3 training."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from data_types import (
    EpisodeResult,
    ActionStatistics,
    TrainingMetrics,
    AggregatedTrainingMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticsSummary:
    """Summary statistics computed from tracked metrics."""

    # Episode statistics
    total_episodes: int
    num_successes: int
    success_rate: float
    mean_reward: float
    max_reward: float
    min_reward: float
    final_50_mean_reward: Optional[float]

    # Action statistics
    mean_main_thruster: Optional[float]
    mean_side_thruster: Optional[float]
    mean_action_magnitude: Optional[float]
    high_thruster_episode_ratio: Optional[float]

    # Training statistics
    mean_q_value: Optional[float]
    q_value_trend: Optional[tuple]  # (first_10_avg, last_10_avg)
    mean_actor_loss: Optional[float]
    mean_critic_loss: Optional[float]
    mean_actor_grad_norm: Optional[float]
    mean_critic_grad_norm: Optional[float]


class DiagnosticsTracker:
    """Collects and stores training metrics without performing I/O.

    This class is responsible for accumulating metrics during training.
    Use DiagnosticsReporter for outputting the collected data.
    """

    def __init__(self) -> None:
        # Episode results
        self.episode_results: List[EpisodeResult] = []
        self.successes: List[int] = []

        # Action statistics per episode
        self.action_stats: List[ActionStatistics] = []

        # Training metrics (recorded periodically, not every episode)
        self.q_values: List[float] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.actor_grad_norms: List[float] = []
        self.critic_grad_norms: List[float] = []

    def record_episode(self, result: EpisodeResult) -> None:
        """Record a completed episode result.

        Args:
            result: The episode result to record
        """
        self.episode_results.append(result)
        if result.success:
            self.successes.append(result.episode_num)

    def record_action_stats(self, stats: ActionStatistics) -> None:
        """Record action statistics for an episode.

        Args:
            stats: Action statistics computed from episode actions
        """
        self.action_stats.append(stats)

    def record_training_metrics(self, metrics: AggregatedTrainingMetrics) -> None:
        """Record aggregated training metrics.

        Args:
            metrics: Aggregated metrics from training updates
        """
        self.q_values.append(metrics.avg_q_value)
        self.actor_losses.append(metrics.actor_loss)
        self.critic_losses.append(metrics.critic_loss)
        self.actor_grad_norms.append(metrics.actor_grad_norm)
        self.critic_grad_norms.append(metrics.critic_grad_norm)

    def get_rewards(self) -> List[float]:
        """Get list of total rewards for all episodes."""
        return [r.total_reward for r in self.episode_results]

    def get_summary(self) -> DiagnosticsSummary:
        """Compute summary statistics from all tracked metrics.

        Returns:
            DiagnosticsSummary with computed statistics
        """
        rewards = self.get_rewards()
        num_episodes = len(rewards)

        if num_episodes == 0:
            return DiagnosticsSummary(
                total_episodes=0, num_successes=0, success_rate=0.0,
                mean_reward=0.0, max_reward=0.0, min_reward=0.0,
                final_50_mean_reward=None, mean_main_thruster=None,
                mean_side_thruster=None, mean_action_magnitude=None,
                high_thruster_episode_ratio=None, mean_q_value=None,
                q_value_trend=None, mean_actor_loss=None,
                mean_critic_loss=None, mean_actor_grad_norm=None,
                mean_critic_grad_norm=None
            )

        # Episode statistics
        final_50_mean = (
            float(np.mean(rewards[-50:]))
            if num_episodes >= 50 else None
        )

        # Action statistics
        mean_main = None
        mean_side = None
        mean_magnitude = None
        high_thruster_ratio = None

        if self.action_stats:
            mean_main = float(np.mean([s.mean_main_thruster for s in self.action_stats]))
            mean_side = float(np.mean([s.mean_side_thruster for s in self.action_stats]))
            mean_magnitude = float(np.mean([s.mean_magnitude for s in self.action_stats]))
            high_thruster_count = sum(
                1 for s in self.action_stats if s.mean_main_thruster > 0.5
            )
            high_thruster_ratio = high_thruster_count / len(self.action_stats)

        # Training statistics
        mean_q = None
        q_trend = None
        mean_actor_loss = None
        mean_critic_loss = None
        mean_actor_grad = None
        mean_critic_grad = None

        if self.q_values:
            mean_q = float(np.mean(self.q_values))
            if len(self.q_values) >= 10:
                q_trend = (
                    float(np.mean(self.q_values[:10])),
                    float(np.mean(self.q_values[-10:]))
                )
            mean_actor_loss = float(np.mean(self.actor_losses))
            mean_critic_loss = float(np.mean(self.critic_losses))
            mean_actor_grad = float(np.mean(self.actor_grad_norms))
            mean_critic_grad = float(np.mean(self.critic_grad_norms))

        return DiagnosticsSummary(
            total_episodes=num_episodes,
            num_successes=len(self.successes),
            success_rate=len(self.successes) / num_episodes,
            mean_reward=float(np.mean(rewards)),
            max_reward=float(np.max(rewards)),
            min_reward=float(np.min(rewards)),
            final_50_mean_reward=final_50_mean,
            mean_main_thruster=mean_main,
            mean_side_thruster=mean_side,
            mean_action_magnitude=mean_magnitude,
            high_thruster_episode_ratio=high_thruster_ratio,
            mean_q_value=mean_q,
            q_value_trend=q_trend,
            mean_actor_loss=mean_actor_loss,
            mean_critic_loss=mean_critic_loss,
            mean_actor_grad_norm=mean_actor_grad,
            mean_critic_grad_norm=mean_critic_grad
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert all tracked data to a dictionary for serialization."""
        return {
            'episode_results': [asdict(r) for r in self.episode_results],
            'successes': self.successes,
            'action_stats': [asdict(s) for s in self.action_stats],
            'q_values': self.q_values,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'actor_grad_norms': self.actor_grad_norms,
            'critic_grad_norms': self.critic_grad_norms
        }


class DiagnosticsReporter:
    """Handles output of diagnostics to various destinations.

    Separates I/O concerns from data collection.
    """

    def __init__(self, tracker: DiagnosticsTracker) -> None:
        self.tracker = tracker

    def print_summary(self) -> None:
        """Print comprehensive diagnostic summary to console."""
        summary = self.tracker.get_summary()

        print("\n" + "=" * 80)
        print("TRAINING DIAGNOSTICS SUMMARY")
        print("=" * 80)

        # Reward statistics
        print(f"\n--- REWARD STATISTICS ---")
        print(f"Total episodes: {summary.total_episodes}")
        print(f"Successes: {summary.num_successes} (episodes: {self.tracker.successes})")
        print(f"Success rate: {summary.success_rate * 100:.1f}%")
        print(f"Mean reward: {summary.mean_reward:.2f}")
        print(f"Max reward: {summary.max_reward:.2f}")
        print(f"Min reward: {summary.min_reward:.2f}")

        if summary.final_50_mean_reward is not None:
            print(f"Final 50 episodes mean reward: {summary.final_50_mean_reward:.2f}")

        # Action statistics
        print(f"\n--- ACTION STATISTICS ---")
        if summary.mean_main_thruster is not None:
            print(f"Episodes with action tracking: {len(self.tracker.action_stats)}")
            print(f"Mean main thruster: {summary.mean_main_thruster:.3f}")
            print(f"Mean side thruster: {summary.mean_side_thruster:.3f}")
            print(f"Mean action magnitude: {summary.mean_action_magnitude:.3f}")

            if len(self.tracker.action_stats) >= 50:
                recent_main = np.mean([
                    s.mean_main_thruster for s in self.tracker.action_stats[-50:]
                ])
                recent_side = np.mean([
                    s.mean_side_thruster for s in self.tracker.action_stats[-50:]
                ])
                recent_mag = np.mean([
                    s.mean_magnitude for s in self.tracker.action_stats[-50:]
                ])
                print(f"\nLast 50 episodes:")
                print(f"  Main thruster: {recent_main:.3f}")
                print(f"  Side thruster: {recent_side:.3f}")
                print(f"  Action magnitude: {recent_mag:.3f}")

            high_count = int(summary.high_thruster_episode_ratio * len(self.tracker.action_stats))
            print(f"\nEpisodes with high main thruster (>0.5): "
                  f"{high_count}/{len(self.tracker.action_stats)}")
        else:
            print("No action data collected (training hasn't started yet)")

        # Training metrics
        print(f"\n--- TRAINING METRICS ---")
        if summary.mean_q_value is not None:
            print(f"Mean Q-value: {summary.mean_q_value:.3f}")
            if summary.q_value_trend:
                print(f"Q-value trend (first 10 vs last 10): "
                      f"{summary.q_value_trend[0]:.3f} -> {summary.q_value_trend[1]:.3f}")
            print(f"Mean actor loss: {summary.mean_actor_loss:.4f}")
            print(f"Mean critic loss: {summary.mean_critic_loss:.4f}")
            print(f"Mean actor gradient norm: {summary.mean_actor_grad_norm:.4f}")
            print(f"Mean critic gradient norm: {summary.mean_critic_grad_norm:.4f}")

            high_loss_count = sum(1 for x in self.tracker.actor_losses if x > 1.0)
            print(f"\nEpisodes with high actor loss (>1.0): "
                  f"{high_loss_count}/{len(self.tracker.actor_losses)}")
        else:
            print("No training metrics collected yet")

        # Recent episodes
        self._print_recent_episodes()

        # Key data
        self._print_key_data()

        print("\n" + "=" * 80)
        print("END OF DIAGNOSTICS")
        print("=" * 80)

    def _print_recent_episodes(self, n: int = 5) -> None:
        """Print details of recent episodes."""
        print(f"\n--- LAST {n} EPISODES DETAIL ---")

        results = self.tracker.episode_results
        if not results:
            print("No episodes recorded")
            return

        start_idx = max(0, len(results) - n)
        for result in results[start_idx:]:
            info_str = str(result)

            # Try to find corresponding action stats
            stats_offset = len(results) - len(self.tracker.action_stats)
            stats_idx = result.episode_num - stats_offset

            if 0 <= stats_idx < len(self.tracker.action_stats):
                stats = self.tracker.action_stats[stats_idx]
                q_idx = stats_idx if stats_idx < len(self.tracker.q_values) else -1
                q_val = self.tracker.q_values[q_idx] if q_idx >= 0 else 0

                info_str += (f", Main: {stats.mean_main_thruster:.2f}, "
                            f"Side: {stats.mean_side_thruster:.2f}, Q: {q_val:.2f}")

            print(info_str)

    def _print_key_data(self) -> None:
        """Print raw data lists for analysis."""
        print("\n" + "=" * 80)
        print("KEY DATA FOR ANALYSIS")
        print("=" * 80)

        rewards = self.tracker.get_rewards()
        print(f"\nReward list (last 50): {rewards[-50:]}")

        if self.tracker.action_stats:
            main_thrusters = [s.mean_main_thruster for s in self.tracker.action_stats]
            print(f"\nMain thruster list (last 50): {main_thrusters[-50:]}")

        if self.tracker.q_values:
            print(f"\nQ-values list (last 50): {self.tracker.q_values[-50:]}")
            print(f"\nActor losses list (last 50): {self.tracker.actor_losses[-50:]}")

    def save_to_json(self, path: Path) -> None:
        """Save all tracked data to a JSON file.

        Args:
            path: Path to save the JSON file
        """
        data = self.tracker.to_dict()
        data['summary'] = asdict(self.tracker.get_summary())

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Diagnostics saved to {path}")

    def log_episode(self, result: EpisodeResult, noise_scale: float) -> None:
        """Log a single episode result.

        Args:
            result: The episode result
            noise_scale: Current noise scale for exploration
        """
        status = "SUCCESS" if result.success else "FAILURE"
        logger.info(
            f"Episode {result.episode_num}: {status}, "
            f"Reward: {result.total_reward:.1f} "
            f"(env: {result.env_reward:.1f}, shaped: {result.shaped_bonus:.1f})"
        )

    def log_training_update(
        self,
        metrics: AggregatedTrainingMetrics,
        noise_scale: float
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Aggregated training metrics
            noise_scale: Current noise scale
        """
        logger.info(
            f"Training update - Critic Loss: {metrics.critic_loss:.4f}, "
            f"Actor Loss: {metrics.actor_loss:.4f}, "
            f"Avg Q: {metrics.avg_q_value:.3f}, "
            f"Noise: {noise_scale:.3f}"
        )
