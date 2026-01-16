"""Output mode abstraction for controlling print verbosity.

This module provides a centralized system for controlling output verbosity
across the simulation. It supports three modes:
- HUMAN: Verbose, narrative, detailed output with emojis
- AGENT: Minimal, low-entropy, structured output for LLM context efficiency
- SILENT: No console output

The simulation behavior is identical across all modes - only output differs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, List, Dict, Any

if TYPE_CHECKING:
    from analysis.behavior_analysis import BehaviorReport


class PrintMode(Enum):
    """Output verbosity modes."""
    HUMAN = "human"    # Verbose, narrative output with emojis
    AGENT = "agent"    # Minimal, structured output for LLMs
    SILENT = "silent"  # No console output


@dataclass
class AgentModeStats:
    """Incremental statistics for agent-mode periodic summaries.

    Tracks rolling statistics that are reset after each periodic summary.
    Uses exponential moving averages and simple counters for efficiency.
    """
    # Episode counters (reset each summary period)
    episodes_since_summary: int = 0
    successes_since_summary: int = 0

    # Reward accumulators (reset each summary period)
    reward_sum: float = 0.0
    reward_min: float = float('inf')
    reward_max: float = float('-inf')

    # Outcome counters (reset each summary period)
    landed_count: int = 0
    crashed_count: int = 0
    timed_out_count: int = 0

    # Global trackers (never reset)
    total_episodes: int = 0
    total_successes: int = 0
    current_streak: int = 0
    max_streak: int = 0

    # Exponential moving averages (alpha=0.1 for smoothing)
    ema_reward: Optional[float] = None
    ema_success_rate: Optional[float] = None

    def record_episode(
        self,
        env_reward: float,
        success: bool,
        outcome_category: str
    ) -> None:
        """Record a single episode's data.

        Args:
            env_reward: Raw environment reward
            success: Whether episode achieved success threshold
            outcome_category: One of 'landed', 'crashed', 'timed_out', 'flew_off'
        """
        # Period counters
        self.episodes_since_summary += 1
        if success:
            self.successes_since_summary += 1

        # Reward stats
        self.reward_sum += env_reward
        self.reward_min = min(self.reward_min, env_reward)
        self.reward_max = max(self.reward_max, env_reward)

        # Outcome counters
        if outcome_category == 'landed':
            self.landed_count += 1
        elif outcome_category == 'crashed':
            self.crashed_count += 1
        elif outcome_category == 'timed_out':
            self.timed_out_count += 1

        # Global trackers
        self.total_episodes += 1
        if success:
            self.total_successes += 1
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)
        else:
            self.current_streak = 0

        # Update EMAs (alpha=0.1)
        alpha = 0.1
        if self.ema_reward is None:
            self.ema_reward = env_reward
        else:
            self.ema_reward = alpha * env_reward + (1 - alpha) * self.ema_reward

        success_val = 1.0 if success else 0.0
        if self.ema_success_rate is None:
            self.ema_success_rate = success_val
        else:
            self.ema_success_rate = alpha * success_val + (1 - alpha) * self.ema_success_rate

    def get_period_stats(self) -> Dict[str, Any]:
        """Get statistics for the current period and reset period counters.

        Returns:
            Dictionary with period statistics
        """
        n = self.episodes_since_summary
        if n == 0:
            return {}

        stats = {
            'n': n,
            'success_rate': self.successes_since_summary / n * 100,
            'mean_reward': self.reward_sum / n,
            'min_reward': self.reward_min if self.reward_min != float('inf') else 0,
            'max_reward': self.reward_max if self.reward_max != float('-inf') else 0,
            'landed_pct': self.landed_count / n * 100,
            'crashed_pct': self.crashed_count / n * 100,
            'timed_out_pct': self.timed_out_count / n * 100,
            'total_episodes': self.total_episodes,
            'total_success_rate': self.total_successes / self.total_episodes * 100,
            'current_streak': self.current_streak,
            'max_streak': self.max_streak,
            'ema_reward': self.ema_reward,
            'ema_success_rate': self.ema_success_rate * 100 if self.ema_success_rate else 0,
        }

        # Reset period counters
        self.episodes_since_summary = 0
        self.successes_since_summary = 0
        self.reward_sum = 0.0
        self.reward_min = float('inf')
        self.reward_max = float('-inf')
        self.landed_count = 0
        self.crashed_count = 0
        self.timed_out_count = 0

        return stats


class OutputController:
    """Centralized controller for all simulation output.

    Provides mode-aware methods for printing episode status, diagnostics,
    and periodic summaries. All print decisions go through this class.
    """

    def __init__(self, mode: PrintMode = PrintMode.HUMAN):
        self.mode = mode
        self.agent_stats = AgentModeStats()
        self._summary_interval = 100  # Episodes between agent-mode summaries

    @classmethod
    def from_string(cls, mode_str: str) -> 'OutputController':
        """Create OutputController from string mode name.

        Args:
            mode_str: One of 'human', 'agent', 'silent'

        Returns:
            OutputController configured for the specified mode
        """
        mode_map = {
            'human': PrintMode.HUMAN,
            'agent': PrintMode.AGENT,
            'silent': PrintMode.SILENT,
        }
        mode = mode_map.get(mode_str.lower(), PrintMode.HUMAN)
        return cls(mode)

    def print_episode(
        self,
        episode_num: int,
        success: bool,
        outcome: str,
        outcome_category: str,
        env_reward: float,
        shaped_bonus: float,
        total_reward: float,
        rendered: bool,
        behavior_report: Optional['BehaviorReport'] = None
    ) -> None:
        """Print episode status based on current mode.

        Args:
            episode_num: Episode number
            success: Whether episode was successful
            outcome: Outcome string (e.g., 'LANDED_PERFECTLY')
            outcome_category: Category ('landed', 'crashed', etc.)
            env_reward: Raw environment reward
            shaped_bonus: Reward shaping bonus
            total_reward: Total reward (env + shaped)
            rendered: Whether episode was rendered
            behavior_report: Optional behavior report for detailed output
        """
        if self.mode == PrintMode.SILENT:
            return

        if self.mode == PrintMode.AGENT:
            # Minimal output: just episode number
            print(f"Run: {episode_num}")
            # Record stats for periodic summary
            self.agent_stats.record_episode(env_reward, success, outcome_category)
            return

        # HUMAN mode: full verbose output
        from constants import SAFE_LANDING_OUTCOMES
        from analysis.output_formatter import format_behavior_output

        status_icon = '\u2713' if success else '\u2717'
        rendered_tag = ' \U0001F3AC' if rendered else ''
        landed_safely = outcome in SAFE_LANDING_OUTCOMES
        landing_indicator = '\u2705 Landed Safely' if landed_safely else '\u274C Didn\'t land safely'

        print(f"Run {episode_num} {status_icon} {outcome}{rendered_tag} {landing_indicator} \U0001F955 Reward: {total_reward:.1f} (env: {env_reward:.1f} / shaped: {shaped_bonus:+.1f})")

        # Print categorized behaviors
        if behavior_report and behavior_report.behaviors:
            print(format_behavior_output(behavior_report))

    def should_print_periodic_summary(self, episode_num: int) -> bool:
        """Check if a periodic summary should be printed.

        Args:
            episode_num: Current episode number

        Returns:
            True if summary should be printed
        """
        if self.mode == PrintMode.SILENT:
            return False
        return episode_num > 0 and episode_num % self._summary_interval == 0

    def print_periodic_summary(self, episode_num: int) -> None:
        """Print a periodic summary based on current mode.

        For HUMAN mode, this is handled by reporter.print_summary().
        For AGENT mode, this prints a compact structured block.

        Args:
            episode_num: Current episode number
        """
        if self.mode == PrintMode.SILENT:
            return

        if self.mode == PrintMode.AGENT:
            stats = self.agent_stats.get_period_stats()
            if not stats:
                return

            # Compact structured output for agent consumption
            print(f"--- Summary @ {episode_num} ---")
            print(f"Period: n={stats['n']} succ={stats['success_rate']:.0f}% reward={stats['mean_reward']:.0f}[{stats['min_reward']:.0f},{stats['max_reward']:.0f}]")
            print(f"Outcomes: land={stats['landed_pct']:.0f}% crash={stats['crashed_pct']:.0f}% timeout={stats['timed_out_pct']:.0f}%")
            print(f"Overall: {stats['total_episodes']}ep succ={stats['total_success_rate']:.0f}% streak={stats['current_streak']}/{stats['max_streak']}")
            print(f"EMA: reward={stats['ema_reward']:.0f} succ={stats['ema_success_rate']:.0f}%")
            return

        # HUMAN mode: handled externally by reporter.print_summary()
        pass

    def print_training_started(self, episode_num: int, buffer_size: int) -> None:
        """Print training started message.

        Args:
            episode_num: Episode when training started
            buffer_size: Number of experiences in buffer
        """
        if self.mode == PrintMode.SILENT:
            return

        if self.mode == PrintMode.AGENT:
            print(f"Training started @ {episode_num} ({buffer_size} exp)")
            return

        # HUMAN mode: use logger (handled externally)
        pass

    def print_final_summary(
        self,
        completed_episodes: int,
        error_occurred: Optional[str],
        elapsed_time: Optional[float],
        total_steps: int,
        training_steps: int
    ) -> None:
        """Print final training summary.

        Args:
            completed_episodes: Total episodes completed
            error_occurred: Error message if training terminated early
            elapsed_time: Total elapsed time in seconds
            total_steps: Total environment steps
            training_steps: Total training updates
        """
        if self.mode == PrintMode.SILENT:
            return

        if self.mode == PrintMode.AGENT:
            status = "ERROR" if error_occurred else "DONE"
            print(f"=== {status}: {completed_episodes}ep ===")
            if elapsed_time and elapsed_time > 0:
                sps = total_steps / elapsed_time
                ups = training_steps / elapsed_time
                print(f"Time={elapsed_time:.0f}s SPS={sps:.0f} UPS={ups:.0f}")
            if error_occurred:
                print(f"Error: {error_occurred}")
            return

        # HUMAN mode: verbose output
        print("\n" + "=" * 80)
        if error_occurred:
            print(f"TRAINING TERMINATED EARLY DUE TO ERROR")
            print(f"Error: {error_occurred}")
            print(f"Completed {completed_episodes} episodes before error")
        else:
            print(f"TRAINING COMPLETED SUCCESSFULLY")
            print(f"Completed {completed_episodes} episodes")
        print("=" * 80)

        if elapsed_time is not None and elapsed_time > 0 and total_steps > 0:
            final_sps = total_steps / elapsed_time
            final_ups = training_steps / elapsed_time
            print(f"\nTotal simulation time: {elapsed_time:.2f} seconds")
            print(f"Average speed: {final_sps:.0f} steps/sec | {final_ups:.0f} updates/sec")
            print(f"Total steps: {total_steps:,} | Total updates: {training_steps:,}")

    def print_diagnostics_header(self) -> None:
        """Print diagnostics header."""
        if self.mode != PrintMode.HUMAN:
            return
        print("\n" + "=" * 80)
        print("TRAINING DIAGNOSTICS SUMMARY")
        print("=" * 80)

    def print_diagnostics_footer(self) -> None:
        """Print diagnostics footer."""
        if self.mode != PrintMode.HUMAN:
            return
        print("\n" + "=" * 80)
        print("END OF DIAGNOSTICS")
        print("=" * 80)

    def is_verbose(self) -> bool:
        """Check if verbose output is enabled (HUMAN mode)."""
        return self.mode == PrintMode.HUMAN

    def is_silent(self) -> bool:
        """Check if all output is suppressed."""
        return self.mode == PrintMode.SILENT

    def is_agent_mode(self) -> bool:
        """Check if agent (minimal) mode is active."""
        return self.mode == PrintMode.AGENT
