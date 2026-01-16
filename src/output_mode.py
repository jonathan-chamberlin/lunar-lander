"""Output mode abstraction for controlling print verbosity.

This module provides a centralized system for controlling output verbosity
across the simulation. It supports three modes:
- HUMAN: Verbose, narrative, detailed output with emojis
- MINIMAL: Minimal, low-entropy, structured output for LLM context efficiency
- SILENT: No console output

The simulation behavior is identical across all modes - only output differs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, List, Dict, Any

if TYPE_CHECKING:
    from analysis.behavior_analysis import BehaviorReport
    from analysis.diagnostics import DiagnosticsTracker


class PrintMode(Enum):
    """Output verbosity modes."""
    HUMAN = "human"        # Verbose, narrative output with emojis
    MINIMAL = "minimal"    # Minimal, structured output for LLMs
    SILENT = "silent"      # No console output
    BACKGROUND = "background"  # Minimal: only batch completion messages


@dataclass
class CheckpointSnapshot:
    """Snapshot of key metrics at a checkpoint for delta calculation."""
    episode: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    landed_pct: float = 0.0
    crashed_pct: float = 0.0
    max_streak: int = 0

    # Quality metrics
    upright_rate: float = 0.0
    centered_rate: float = 0.0
    controlled_descent_rate: float = 0.0
    contact_rate: float = 0.0
    clean_touchdown_rate: float = 0.0

    # Training metrics
    mean_q_value: float = 0.0
    actor_loss: float = 0.0
    critic_loss: float = 0.0


@dataclass
class MinimalModeStats:
    """Incremental statistics for minimal-mode periodic summaries.

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

    # Previous checkpoint for delta calculation
    prev_checkpoint: Optional[CheckpointSnapshot] = None

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


def _fmt_delta(current: float, previous: float, fmt: str = ".0f", suffix: str = "") -> str:
    """Format a delta value with sign indicator.

    Args:
        current: Current value
        previous: Previous value
        fmt: Format string for the delta
        suffix: Suffix to add (e.g., '%')

    Returns:
        Formatted delta string like "(+5%)" or "(-3)"
    """
    delta = current - previous
    if abs(delta) < 0.5:  # Effectively no change
        return ""
    sign = "+" if delta > 0 else ""
    return f"({sign}{delta:{fmt}}{suffix})"


def _fmt_trend(current: float, previous: float) -> str:
    """Get trend arrow based on change direction."""
    delta = current - previous
    if delta > 2:
        return "^"
    elif delta < -2:
        return "v"
    return "="


class OutputController:
    """Centralized controller for all simulation output.

    Provides mode-aware methods for printing episode status, diagnostics,
    and periodic summaries. All print decisions go through this class.
    """

    def __init__(self, mode: PrintMode = PrintMode.HUMAN):
        self.mode = mode
        self.minimal_stats = MinimalModeStats()
        self._summary_interval = 100  # Episodes between minimal-mode summaries

    @classmethod
    def from_string(cls, mode_str: str) -> 'OutputController':
        """Create OutputController from string mode name.

        Args:
            mode_str: One of 'human', 'minimal', 'silent'

        Returns:
            OutputController configured for the specified mode
        """
        mode_map = {
            'human': PrintMode.HUMAN,
            'minimal': PrintMode.MINIMAL,
            'silent': PrintMode.SILENT,
            'background': PrintMode.BACKGROUND,
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

        if self.mode == PrintMode.BACKGROUND:
            # No per-episode output, but still track stats for batch summary
            self.minimal_stats.record_episode(env_reward, success, outcome_category)
            return

        if self.mode == PrintMode.MINIMAL:
            # Minimal output: just episode number
            print(f"Run: {episode_num}")
            # Record stats for periodic summary
            self.minimal_stats.record_episode(env_reward, success, outcome_category)
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

    def print_batch_completed(
        self,
        batch_num: int,
        batch_start: int,
        batch_end: int,
        tracker: Optional['DiagnosticsTracker'] = None
    ) -> None:
        """Print batch completion message for BACKGROUND mode.

        Args:
            batch_num: Batch number (1-indexed)
            batch_start: First episode in batch
            batch_end: Last episode in batch
            tracker: DiagnosticsTracker for success rate (optional)
        """
        if self.mode != PrintMode.BACKGROUND:
            return

        # Get period stats and compute success rate for this batch
        period_stats = self.minimal_stats.get_period_stats()
        success_pct = period_stats.get('success_rate', 0) if period_stats else 0

        print(f"Batch {batch_num} (Runs {batch_start}-{batch_end}) completed. Success: {success_pct:.0f}%")

    def print_periodic_summary(
        self,
        episode_num: int,
        tracker: Optional['DiagnosticsTracker'] = None
    ) -> None:
        """Print a periodic summary based on current mode.

        For HUMAN mode, this is handled by reporter.print_summary().
        For MINIMAL mode, this prints a detailed structured block with deltas.
        For BACKGROUND mode, this is handled by print_batch_completed().

        Args:
            episode_num: Current episode number
            tracker: DiagnosticsTracker for detailed statistics (optional for minimal mode)
        """
        if self.mode == PrintMode.SILENT:
            return

        if self.mode == PrintMode.BACKGROUND:
            # BACKGROUND mode: batch completion handled separately via print_batch_completed
            return

        if self.mode == PrintMode.MINIMAL:
            self._print_minimal_detailed_summary(episode_num, tracker)
            return

        # HUMAN mode: handled externally by reporter.print_summary()
        pass

    def _print_minimal_detailed_summary(
        self,
        episode_num: int,
        tracker: Optional['DiagnosticsTracker'] = None
    ) -> None:
        """Print detailed minimal mode summary with tiered structure and deltas.

        Args:
            episode_num: Current episode number
            tracker: DiagnosticsTracker for detailed statistics
        """
        period_stats = self.minimal_stats.get_period_stats()
        if not period_stats:
            return

        prev = self.minimal_stats.prev_checkpoint

        # === QUICK SUMMARY (4 lines) ===
        print(f"--- Summary @ {episode_num} ---")
        print(f"Period: n={period_stats['n']} succ={period_stats['success_rate']:.0f}% reward={period_stats['mean_reward']:.0f}[{period_stats['min_reward']:.0f},{period_stats['max_reward']:.0f}]")
        print(f"Outcomes: land={period_stats['landed_pct']:.0f}% crash={period_stats['crashed_pct']:.0f}% timeout={period_stats['timed_out_pct']:.0f}%")
        print(f"Overall: {period_stats['total_episodes']}ep succ={period_stats['total_success_rate']:.0f}% streak={period_stats['current_streak']}/{period_stats['max_streak']}")
        print(f"EMA: reward={period_stats['ema_reward']:.0f} succ={period_stats['ema_success_rate']:.0f}%")

        # === DETAILED SECTIONS (from tracker if available) ===
        if tracker is None:
            self._save_checkpoint(episode_num, period_stats, None)
            return

        summary = tracker.get_summary()
        adv_stats = tracker.get_advanced_statistics()
        behavior_stats = tracker.get_behavior_statistics()
        streak_stats = tracker.get_streak_statistics()

        # [METRICS] section
        print(f"\n[METRICS]")

        # Reward metrics with deltas
        reward_delta = _fmt_delta(summary.mean_reward, prev.mean_reward if prev else 0, ".0f") if prev else ""
        print(f"reward: mean={summary.mean_reward:.0f}{reward_delta} std={adv_stats.get('env_reward', {}).get('std', 0):.0f} max={summary.max_reward:.0f} min={summary.min_reward:.0f}", end="")
        if summary.final_50_mean_reward is not None:
            print(f" l50={summary.final_50_mean_reward:.0f}")
        else:
            print()

        # Success metrics
        first_success = adv_stats.get('first_success_episode', 'N/A')
        near_misses = adv_stats.get('near_misses', {}).get('count', 0)
        succ_delta = _fmt_delta(summary.success_rate * 100, prev.success_rate if prev else 0, ".0f", "%") if prev else ""
        print(f"success: n={summary.num_successes} rate={summary.success_rate*100:.0f}%{succ_delta} first@ep{first_success} nearMiss={near_misses}")

        # Streak metrics
        streak_delta = _fmt_delta(streak_stats['max_streak'], prev.max_streak if prev else 0, ".0f") if prev else ""
        print(f"streaks: cur={streak_stats['current_streak']} max={streak_stats['max_streak']}{streak_delta}@ep{streak_stats['max_streak_episode']}")

        # [OUTCOMES] section - show top outcome types
        print(f"\n[OUTCOMES]")
        if behavior_stats:
            sorted_outcomes = sorted(behavior_stats.outcome_counts.items(), key=lambda x: -x[1])
            outcome_parts = [f"{o}={c}" for o, c in sorted_outcomes[:6]]
            print(" ".join(outcome_parts))

            # Crash breakdown if crashes exist
            if behavior_stats.crash_type_distribution:
                crash_parts = [f"{t}={p:.0f}%" for t, p in sorted(
                    behavior_stats.crash_type_distribution.items(), key=lambda x: -x[1]
                )[:4]]
                print(f"crashes: {' '.join(crash_parts)}")

        # [QUALITY] section
        print(f"\n[QUALITY]")
        if behavior_stats:
            # Flight quality with deltas
            upright = behavior_stats.stayed_upright_rate
            centered = behavior_stats.stayed_centered_rate
            ctrl_desc = behavior_stats.controlled_descent_rate
            ctrl_thru = behavior_stats.controlled_throughout_rate

            upright_d = _fmt_delta(upright, prev.upright_rate if prev else 0, ".0f", "%") if prev else ""
            centered_d = _fmt_delta(centered, prev.centered_rate if prev else 0, ".0f", "%") if prev else ""
            ctrl_desc_d = _fmt_delta(ctrl_desc, prev.controlled_descent_rate if prev else 0, ".0f", "%") if prev else ""

            print(f"upright={upright:.0f}%{upright_d} centered={centered:.0f}%{centered_d} ctrlDesc={ctrl_desc:.0f}%{ctrl_desc_d} ctrlThru={ctrl_thru:.0f}%")

            # Progress indicators with deltas
            low_alt = behavior_stats.low_altitude_rate
            contact = behavior_stats.contact_rate
            clean_td = behavior_stats.clean_touchdown_rate

            contact_d = _fmt_delta(contact, prev.contact_rate if prev else 0, ".0f", "%") if prev else ""
            clean_td_d = _fmt_delta(clean_td, prev.clean_touchdown_rate if prev else 0, ".0f", "%") if prev else ""

            print(f"lowAlt={low_alt:.0f}% contact={contact:.0f}%{contact_d} cleanTD={clean_td:.0f}%{clean_td_d}")
            print(f"goodBehavior={behavior_stats.good_behavior_rate:.0f}% badBehavior={behavior_stats.bad_behavior_rate:.0f}%")

        # [TRAINING] section
        print(f"\n[TRAINING]")
        if summary.mean_q_value is not None:
            q_delta = _fmt_delta(summary.mean_q_value, prev.mean_q_value if prev else 0, ".1f") if prev else ""
            q_trend = ""
            if summary.q_value_trend:
                q_trend = f"(trend:{summary.q_value_trend[0]:.1f}->{summary.q_value_trend[1]:.1f})"
            print(f"Q={summary.mean_q_value:.1f}{q_delta}{q_trend} aLoss={summary.mean_actor_loss:.3f} cLoss={summary.mean_critic_loss:.3f}")
            print(f"aGrad={summary.mean_actor_grad_norm:.2f} cGrad={summary.mean_critic_grad_norm:.2f} highLoss={sum(1 for x in tracker.actor_losses if x > 1.0)}/{len(tracker.actor_losses)}")
        else:
            print("No training data yet")

        # [BATCH_PROGRESS] section - show per-batch trends
        print(f"\n[BATCH_PROGRESS]")
        if behavior_stats and len(behavior_stats.batch_success_rates) > 0:
            # Show batch-by-batch progress (50 episodes each)
            batch_size = 50
            num_batches = len(behavior_stats.batch_success_rates)

            for i in range(num_batches):
                batch_start = i * batch_size + 1
                batch_end = min((i + 1) * batch_size, behavior_stats.total_episodes)
                succ = behavior_stats.batch_success_rates[i]
                outcome_dist = behavior_stats.batch_outcome_distributions[i]
                landed = outcome_dist.get('landed', 0)
                crashed = outcome_dist.get('crashed', 0)

                # Include speed metrics if available
                speed_str = ""
                if i < len(tracker.batch_speed_metrics):
                    speed = tracker.batch_speed_metrics[i]
                    speed_str = f" SPS={speed.sps:.0f}"

                print(f"{batch_start}-{batch_end}: succ={succ:.0f}% land={landed:.0f}% crash={crashed:.0f}%{speed_str}")

        # [DELTA] section - highlight significant changes
        if prev:
            print(f"\n[DELTA from @{prev.episode}]")
            improvements = []
            regressions = []

            # Check each metric for significant change (>3% or >3 units)
            checks = [
                ('success', summary.success_rate * 100, prev.success_rate, '%'),
                ('reward', summary.mean_reward, prev.mean_reward, ''),
                ('landed', period_stats['landed_pct'], prev.landed_pct, '%'),
                ('crashed', period_stats['crashed_pct'], prev.crashed_pct, '%'),  # Lower is better
            ]

            if behavior_stats:
                checks.extend([
                    ('upright', behavior_stats.stayed_upright_rate, prev.upright_rate, '%'),
                    ('centered', behavior_stats.stayed_centered_rate, prev.centered_rate, '%'),
                    ('ctrlDesc', behavior_stats.controlled_descent_rate, prev.controlled_descent_rate, '%'),
                    ('contact', behavior_stats.contact_rate, prev.contact_rate, '%'),
                    ('cleanTD', behavior_stats.clean_touchdown_rate, prev.clean_touchdown_rate, '%'),
                ])

            for name, current, previous, suffix in checks:
                delta = current - previous
                threshold = 3 if suffix == '%' else 5

                # For 'crashed', improvement is negative delta
                is_better = delta < -threshold if name == 'crashed' else delta > threshold
                is_worse = delta > threshold if name == 'crashed' else delta < -threshold

                if is_better:
                    sign = "-" if name == 'crashed' else "+"
                    improvements.append(f"{name}{sign}{abs(delta):.0f}{suffix}")
                elif is_worse:
                    sign = "+" if name == 'crashed' else ""
                    regressions.append(f"{name}{sign}{delta:.0f}{suffix}")

            if improvements:
                print(f"IMPROVED: {' '.join(improvements)}")
            if regressions:
                print(f"REGRESSED: {' '.join(regressions)}")
            if not improvements and not regressions:
                print("STABLE: no significant changes")

        # Save current as checkpoint for next comparison
        self._save_checkpoint(episode_num, period_stats, tracker)

    def _save_checkpoint(
        self,
        episode_num: int,
        period_stats: Dict[str, Any],
        tracker: Optional['DiagnosticsTracker']
    ) -> None:
        """Save current metrics as checkpoint for delta calculation.

        Args:
            episode_num: Current episode number
            period_stats: Period statistics from minimal_stats
            tracker: DiagnosticsTracker for additional metrics
        """
        checkpoint = CheckpointSnapshot(episode=episode_num)

        # Basic metrics from period stats
        checkpoint.success_rate = period_stats.get('total_success_rate', 0)
        checkpoint.mean_reward = period_stats.get('mean_reward', 0)
        checkpoint.landed_pct = period_stats.get('landed_pct', 0)
        checkpoint.crashed_pct = period_stats.get('crashed_pct', 0)
        checkpoint.max_streak = period_stats.get('max_streak', 0)

        if tracker:
            summary = tracker.get_summary()
            behavior_stats = tracker.get_behavior_statistics()

            checkpoint.mean_reward = summary.mean_reward

            if summary.mean_q_value is not None:
                checkpoint.mean_q_value = summary.mean_q_value
                checkpoint.actor_loss = summary.mean_actor_loss
                checkpoint.critic_loss = summary.mean_critic_loss

            if behavior_stats:
                checkpoint.upright_rate = behavior_stats.stayed_upright_rate
                checkpoint.centered_rate = behavior_stats.stayed_centered_rate
                checkpoint.controlled_descent_rate = behavior_stats.controlled_descent_rate
                checkpoint.contact_rate = behavior_stats.contact_rate
                checkpoint.clean_touchdown_rate = behavior_stats.clean_touchdown_rate

        self.minimal_stats.prev_checkpoint = checkpoint

    def print_training_started(self, episode_num: int, buffer_size: int) -> None:
        """Print training started message.

        Args:
            episode_num: Episode when training started
            buffer_size: Number of experiences in buffer
        """
        if self.mode == PrintMode.SILENT:
            return

        if self.mode == PrintMode.MINIMAL:
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
        training_steps: int,
        tracker: Optional['DiagnosticsTracker'] = None
    ) -> None:
        """Print final training summary.

        Args:
            completed_episodes: Total episodes completed
            error_occurred: Error message if training terminated early
            elapsed_time: Total elapsed time in seconds
            total_steps: Total environment steps
            training_steps: Total training updates
            tracker: DiagnosticsTracker for detailed final stats (optional)
        """
        if self.mode == PrintMode.SILENT:
            return

        if self.mode == PrintMode.BACKGROUND:
            # Minimal final summary for background mode
            status = "ERROR" if error_occurred else "DONE"
            success_rate = 0.0
            if tracker and tracker.total_episodes > 0:
                success_rate = tracker.success_count / tracker.total_episodes * 100
            print(f"\n=== {status}: {completed_episodes} episodes, {success_rate:.0f}% success ===")
            if elapsed_time and elapsed_time > 0:
                print(f"Time: {elapsed_time:.0f}s")
            if error_occurred:
                print(f"Error: {error_occurred}")
            return

        if self.mode == PrintMode.MINIMAL:
            status = "ERROR" if error_occurred else "DONE"
            print(f"\n=== {status}: {completed_episodes}ep ===")
            if elapsed_time and elapsed_time > 0:
                sps = total_steps / elapsed_time
                ups = training_steps / elapsed_time
                print(f"Time={elapsed_time:.0f}s SPS={sps:.0f} UPS={ups:.0f}")
            if error_occurred:
                print(f"Error: {error_occurred}")

            # Print final detailed summary if tracker available
            if tracker and completed_episodes > 0:
                # Force a final summary print
                self._print_minimal_detailed_summary(completed_episodes, tracker)
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

    def is_minimal_mode(self) -> bool:
        """Check if minimal mode is active."""
        return self.mode == PrintMode.MINIMAL

    def is_background_mode(self) -> bool:
        """Check if background (batch-only) mode is active."""
        return self.mode == PrintMode.BACKGROUND
