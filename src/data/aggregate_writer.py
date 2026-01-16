"""Periodic aggregate persistence for simulation data.

Writes aggregate snapshots at regular intervals during training.
Aggregates are DERIVED data - computed from RAW runs.jsonl data.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.diagnostics import DiagnosticsTracker

logger = logging.getLogger(__name__)


class AggregateWriter:
    """Writes periodic aggregate snapshots to the aggregates/ directory.

    Persists computed statistics at regular intervals to enable:
    - Checkpoint recovery (resume from last aggregate if simulation interrupted)
    - Progress monitoring without loading all raw data
    - Experiment-level aggregation across simulations

    Data tiers:
    - RAW: runs.jsonl - immutable per-episode records
    - DERIVED: aggregates/*.json - computed from raw (this class)
    - ARTIFACTS: charts/, text/ - regenerable from derived

    Usage:
        writer = AggregateWriter(sim_dir.aggregates_path)
        writer.maybe_write(completed_episodes, diagnostics)  # Every N episodes
        writer.write_final(diagnostics)  # At simulation end
    """

    def __init__(
        self,
        aggregates_path: Path,
        write_interval: int = 100,
    ) -> None:
        """Initialize the aggregate writer.

        Args:
            aggregates_path: Path to the aggregates/ directory
            write_interval: Write aggregate every N episodes (default 100)
        """
        self._aggregates_path = Path(aggregates_path)
        self._write_interval = write_interval
        self._last_write_episode = 0

    @property
    def aggregates_path(self) -> Path:
        """Return the aggregates directory path."""
        return self._aggregates_path

    @property
    def write_interval(self) -> int:
        """Return the write interval."""
        return self._write_interval

    def maybe_write(
        self,
        completed_episodes: int,
        diagnostics: "DiagnosticsTracker",
    ) -> bool:
        """Write aggregate if we've reached a write interval.

        Args:
            completed_episodes: Number of episodes completed
            diagnostics: DiagnosticsTracker with current statistics

        Returns:
            True if aggregate was written, False otherwise
        """
        if completed_episodes == 0:
            return False

        if completed_episodes % self._write_interval != 0:
            return False

        if completed_episodes <= self._last_write_episode:
            return False

        batch_num = completed_episodes // self._write_interval
        filename = f"batch_{batch_num:04d}.json"
        self._write_aggregate(filename, completed_episodes, diagnostics)
        self._last_write_episode = completed_episodes
        return True

    def write_final(self, diagnostics: "DiagnosticsTracker") -> None:
        """Write final aggregate at simulation end.

        Args:
            diagnostics: DiagnosticsTracker with final statistics
        """
        completed_episodes = diagnostics.total_episodes
        self._write_aggregate("final.json", completed_episodes, diagnostics)
        logger.info(f"Final aggregate written to {self._aggregates_path / 'final.json'}")

    def _write_aggregate(
        self,
        filename: str,
        completed_episodes: int,
        diagnostics: "DiagnosticsTracker",
    ) -> None:
        """Write an aggregate snapshot to file.

        Args:
            filename: Name of the aggregate file
            completed_episodes: Number of episodes in this aggregate
            diagnostics: DiagnosticsTracker with statistics
        """
        aggregate = self._build_aggregate(completed_episodes, diagnostics)
        path = self._aggregates_path / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)

        logger.debug(f"Aggregate written to {path}")

    def _build_aggregate(
        self,
        completed_episodes: int,
        diagnostics: "DiagnosticsTracker",
    ) -> Dict[str, Any]:
        """Build aggregate dictionary from diagnostics.

        Args:
            completed_episodes: Number of episodes completed
            diagnostics: DiagnosticsTracker with statistics

        Returns:
            Dictionary with aggregate data
        """
        # Get summary statistics
        summary = diagnostics.get_summary()

        # Get behavior statistics
        behavior_stats = diagnostics.get_behavior_statistics()

        # Get advanced statistics
        advanced_stats = diagnostics.get_advanced_statistics()

        # Get streak statistics
        streak_stats = diagnostics.get_streak_statistics()

        # Get env_reward distribution
        env_reward_dist = diagnostics.get_env_reward_distribution()

        aggregate = {
            "_metadata": {
                "schema_version": "1.0.0",
                "completed_episodes": completed_episodes,
                "created_at": datetime.now().isoformat(),
            },
            "summary": {
                "total_episodes": summary.total_episodes,
                "num_successes": summary.num_successes,
                "success_rate": summary.success_rate,
                "mean_reward": summary.mean_reward,
                "max_reward": summary.max_reward,
                "min_reward": summary.min_reward,
                "final_50_mean_reward": summary.final_50_mean_reward,
            },
            "training": {
                "mean_q_value": summary.mean_q_value,
                "q_value_trend": summary.q_value_trend,
                "mean_actor_loss": summary.mean_actor_loss,
                "mean_critic_loss": summary.mean_critic_loss,
                "mean_actor_grad_norm": summary.mean_actor_grad_norm,
                "mean_critic_grad_norm": summary.mean_critic_grad_norm,
            },
            "streaks": {
                "max_streak": streak_stats.get("max_streak", 0),
                "max_streak_episode": streak_stats.get("max_streak_episode", 0),
                "current_streak": streak_stats.get("current_streak", 0),
            },
        }

        # Add behavior statistics if available
        if behavior_stats is not None:
            aggregate["behaviors"] = {
                "outcome_counts": behavior_stats.outcome_counts,
                "outcome_category_counts": behavior_stats.outcome_category_counts,
                "top_behaviors": [
                    {"behavior": b, "count": c, "percentage": p}
                    for b, c, p in behavior_stats.top_behaviors
                ],
                "quality_rates": {
                    "good_behavior_rate": behavior_stats.good_behavior_rate,
                    "bad_behavior_rate": behavior_stats.bad_behavior_rate,
                    "low_altitude_rate": behavior_stats.low_altitude_rate,
                    "contact_rate": behavior_stats.contact_rate,
                    "clean_touchdown_rate": behavior_stats.clean_touchdown_rate,
                },
                "flight_quality": {
                    "stayed_upright_rate": behavior_stats.stayed_upright_rate,
                    "stayed_centered_rate": behavior_stats.stayed_centered_rate,
                    "controlled_descent_rate": behavior_stats.controlled_descent_rate,
                    "controlled_throughout_rate": behavior_stats.controlled_throughout_rate,
                    "never_stabilized_rate": behavior_stats.never_stabilized_rate,
                },
                "success_correlation": {
                    "success_behavior_rates": behavior_stats.success_behavior_rates,
                    "failure_behavior_rates": behavior_stats.failure_behavior_rates,
                },
                "crash_type_distribution": behavior_stats.crash_type_distribution,
            }

        # Add advanced statistics
        if advanced_stats:
            aggregate["advanced"] = advanced_stats

        # Add env_reward distribution for landing quality analysis
        if env_reward_dist:
            # Remove raw rewards list to keep aggregate concise
            if "landed" in env_reward_dist:
                dist = env_reward_dist["landed"].copy()
                dist.pop("rewards", None)  # Remove raw data
                aggregate["landing_quality"] = dist

        return aggregate

    def read_aggregate(self, filename: str) -> Optional[Dict[str, Any]]:
        """Read an aggregate file.

        Args:
            filename: Name of the aggregate file to read

        Returns:
            Aggregate dictionary, or None if file doesn't exist
        """
        path = self._aggregates_path / filename
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def read_final(self) -> Optional[Dict[str, Any]]:
        """Read the final aggregate.

        Returns:
            Final aggregate dictionary, or None if not available
        """
        return self.read_aggregate("final.json")

    def list_aggregates(self) -> list:
        """List all aggregate files in order.

        Returns:
            List of aggregate filenames sorted by batch number
        """
        if not self._aggregates_path.exists():
            return []

        files = sorted(self._aggregates_path.glob("*.json"))
        return [f.name for f in files]
