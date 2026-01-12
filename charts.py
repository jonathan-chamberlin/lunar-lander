"""Training visualization charts for Lunar Lander RL.

This module generates charts to visualize training progress and effectiveness.
Single Responsibility: Transform collected metrics into visual charts.
"""

from typing import List, Dict, Tuple, Optional
import logging

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import DiagnosticsTracker

logger = logging.getLogger(__name__)


# Behavior categories for report card
REPORT_CARD_BEHAVIORS = [
    # Good behaviors (should increase)
    'STAYED_UPRIGHT',
    'STAYED_CENTERED',
    'CONTROLLED_DESCENT',
    'CONTROLLED_THROUGHOUT',
    'TOUCHED_DOWN_CLEAN',
    'REACHED_LOW_ALTITUDE',
    # Bad behaviors (should decrease)
    'NEVER_STABILIZED',
    'SPINNING_UNCONTROLLED',
    'FREEFALL',
    'ERRATIC_THRUST',
    'CRASHED_FAST_VERTICAL',
    'FLIPPED_OVER',
]

# Outcome categories for stacked area chart
OUTCOME_CATEGORIES = ['landed', 'crashed', 'timed_out', 'flew_off']
OUTCOME_COLORS = {
    'landed': '#2ecc71',     # Green
    'crashed': '#e74c3c',    # Red
    'timed_out': '#f39c12',  # Yellow/Orange
    'flew_off': '#3498db',   # Blue
}

# Outcome mapping from specific outcomes to categories
OUTCOME_TO_CATEGORY = {
    'LANDED_PERFECTLY': 'landed',
    'LANDED_SOFTLY': 'landed',
    'LANDED_HARD': 'landed',
    'LANDED_TILTED': 'landed',
    'LANDED_ONE_LEG': 'landed',
    'CRASHED_FAST_VERTICAL': 'crashed',
    'CRASHED_FAST_TILTED': 'crashed',
    'CRASHED_SIDEWAYS': 'crashed',
    'CRASHED_SPINNING': 'crashed',
    'TIMED_OUT_HOVERING': 'timed_out',
    'TIMED_OUT_DESCENDING': 'timed_out',
    'TIMED_OUT_ASCENDING': 'timed_out',
    'FLEW_OFF_TOP': 'flew_off',
    'FLEW_OFF_LEFT': 'flew_off',
    'FLEW_OFF_RIGHT': 'flew_off',
    'FLEW_OFF_LEFT_TILTED': 'flew_off',
    'FLEW_OFF_RIGHT_TILTED': 'flew_off',
}


class ChartGenerator:
    """Generates training visualization charts from diagnostics data.

    Single Responsibility: Transform collected metrics into visual charts.
    Does NOT collect data - receives DiagnosticsTracker as dependency.

    Args:
        tracker: DiagnosticsTracker containing collected training data
        batch_size: Number of episodes per batch for aggregation (default: 50)
    """

    def __init__(self, tracker: DiagnosticsTracker, batch_size: int = 50) -> None:
        self.tracker = tracker
        self.batch_size = batch_size

    # =========================================================================
    # Main Public Method
    # =========================================================================

    def generate_all(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """Generate all charts in a single figure with subplots.

        Creates a 2x3 grid of subplots containing all training visualizations.

        Args:
            show: Whether to display the figure with plt.show()
            save_path: Optional path to save the figure as an image
        """
        # Check if we have enough data
        num_episodes = len(self.tracker.episode_results)
        if num_episodes < 10:
            logger.warning(f"Not enough episodes ({num_episodes}) to generate meaningful charts")
            return

        # Create figure with 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Training Progress - {num_episodes} Episodes', fontsize=14, fontweight='bold')

        # Generate each chart
        try:
            self._plot_reward_over_time(axes[0, 0])
        except Exception as e:
            logger.warning(f"Failed to plot reward chart: {e}")
            axes[0, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 0].transAxes)

        try:
            self._plot_success_rate(axes[0, 1])
        except Exception as e:
            logger.warning(f"Failed to plot success rate chart: {e}")
            axes[0, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 1].transAxes)

        try:
            self._plot_outcome_stacked_area(axes[0, 2])
        except Exception as e:
            logger.warning(f"Failed to plot outcome chart: {e}")
            axes[0, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[0, 2].transAxes)

        try:
            self._plot_consecutive_streak(axes[1, 0])
        except Exception as e:
            logger.warning(f"Failed to plot streak chart: {e}")
            axes[1, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 0].transAxes)

        try:
            self._plot_behavior_heatmap(axes[1, 1])
        except Exception as e:
            logger.warning(f"Failed to plot behavior heatmap: {e}")
            axes[1, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 1].transAxes)

        try:
            self._plot_behavior_report_card(axes[1, 2])
        except Exception as e:
            logger.warning(f"Failed to plot report card: {e}")
            axes[1, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[1, 2].transAxes)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save if path provided
        if save_path:
            try:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Charts saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save charts: {e}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

    # =========================================================================
    # Individual Chart Methods
    # =========================================================================

    def _plot_reward_over_time(self, ax: plt.Axes) -> None:
        """Chart 1: Episode reward with rolling average.

        Shows raw episode rewards (transparent) with a bold rolling average line.
        Includes reference lines at y=0 and success threshold.

        Args:
            ax: Matplotlib axes to plot on
        """
        rewards = [r.total_reward for r in self.tracker.episode_results]

        if not rewards:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Episode Reward Over Time')
            return

        episodes = list(range(len(rewards)))
        rolling_avg = self._compute_rolling_average(rewards, window=50)

        # Plot raw rewards (transparent scatter)
        ax.scatter(episodes, rewards, alpha=0.2, s=5, color='blue', label='Raw Reward')

        # Plot rolling average (bold line)
        ax.plot(episodes, rolling_avg, color='blue', linewidth=2, label='50-Episode Average')

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=180, color='green', linestyle='--', alpha=0.7, label='Success Threshold (180)')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Reward Over Time')
        ax.legend(loc='lower right', fontsize=8)

    def _plot_success_rate(self, ax: plt.Axes) -> None:
        """Chart 2: Success rate over time in batches.

        Shows success rate per batch (50 episodes each) as a line chart.
        Includes reference line at 50% threshold.

        Args:
            ax: Matplotlib axes to plot on
        """
        results = self.tracker.episode_results

        if not results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Success Rate by Batch')
            return

        # Compute success rate per batch
        def calc_success_rate(batch):
            successes = sum(1 for r in batch if r.success)
            return (successes / len(batch)) * 100

        labels, success_rates = self._compute_batch_stats(results, calc_success_rate)

        # Plot as line chart with markers
        x_positions = list(range(len(labels)))
        ax.plot(x_positions, success_rates, color='green', linewidth=2, marker='o', markersize=4)
        ax.fill_between(x_positions, success_rates, alpha=0.2, color='green')

        # Reference lines
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Threshold')
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80% Target')

        # X-axis labels (show every Nth label to avoid crowding)
        n_labels = len(labels)
        if n_labels > 20:
            step = max(1, n_labels // 10)
            ax.set_xticks(x_positions[::step])
            ax.set_xticklabels(labels[::step], rotation=45, ha='right', fontsize=7)
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)

        ax.set_xlabel('Batch (Episodes)')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Batch (50 Episodes)')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 105)

    def _plot_outcome_stacked_area(self, ax: plt.Axes) -> None:
        """Chart 3: Outcome distribution as stacked area chart.

        Shows proportion of Landed/Crashed/Timed Out/Flew Off over batches.

        Args:
            ax: Matplotlib axes to plot on
        """
        if not self.tracker.behavior_reports:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Outcome Distribution Over Time')
            return

        outcome_data = self._get_outcome_distribution_per_batch()
        num_batches = len(outcome_data['landed'])

        if num_batches == 0:
            ax.text(0.5, 0.5, 'No batch data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Outcome Distribution Over Time')
            return

        x = np.arange(num_batches)

        # Stack the areas (order: landed at bottom, then crashed, timed_out, flew_off)
        landed = np.array(outcome_data['landed'])
        crashed = np.array(outcome_data['crashed'])
        timed_out = np.array(outcome_data['timed_out'])
        flew_off = np.array(outcome_data['flew_off'])

        ax.stackplot(
            x,
            landed, crashed, timed_out, flew_off,
            labels=['Landed', 'Crashed', 'Timed Out', 'Flew Off'],
            colors=[OUTCOME_COLORS['landed'], OUTCOME_COLORS['crashed'],
                    OUTCOME_COLORS['timed_out'], OUTCOME_COLORS['flew_off']],
            alpha=0.8
        )

        # X-axis labels
        batch_labels = [f"{i*self.batch_size + 1}" for i in range(num_batches)]
        if num_batches > 20:
            step = max(1, num_batches // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(batch_labels[::step], fontsize=7)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(batch_labels, fontsize=7)

        ax.set_xlabel('Batch Start Episode')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Outcome Distribution Over Time')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, num_batches - 1)

    def _plot_behavior_heatmap(self, ax: plt.Axes) -> None:
        """Chart 8: Behavior frequency heatmap over batches.

        Shows behavior frequencies as a heatmap with batches on x-axis
        and behaviors on y-axis. Color intensity = frequency percentage.

        Args:
            ax: Matplotlib axes to plot on
        """
        num_episodes = len(self.tracker.behavior_reports)

        if num_episodes == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Behavior Heatmap')
            return

        num_batches = (num_episodes + self.batch_size - 1) // self.batch_size

        # Get all behavior frequencies per batch
        all_batch_freqs = []
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, num_episodes)
            freqs = self._get_behavior_frequencies(start, end)
            all_batch_freqs.append(freqs)

        # Select behaviors to show (use report card behaviors + any high-frequency ones)
        all_behaviors = set()
        for freqs in all_batch_freqs:
            all_behaviors.update(freqs.keys())

        # Prioritize report card behaviors, then add others by frequency
        selected_behaviors = []
        for b in REPORT_CARD_BEHAVIORS:
            if b in all_behaviors:
                selected_behaviors.append(b)

        # Limit to 15 behaviors for readability
        selected_behaviors = selected_behaviors[:15]

        if not selected_behaviors:
            ax.text(0.5, 0.5, 'No behavior data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Behavior Heatmap')
            return

        # Build heatmap matrix
        heatmap_data = np.zeros((len(selected_behaviors), num_batches))
        for batch_idx, freqs in enumerate(all_batch_freqs):
            for behavior_idx, behavior in enumerate(selected_behaviors):
                heatmap_data[behavior_idx, batch_idx] = freqs.get(behavior, 0)

        # Create heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)

        # Labels
        ax.set_yticks(range(len(selected_behaviors)))
        ax.set_yticklabels(selected_behaviors, fontsize=7)

        # X-axis labels (batch numbers)
        if num_batches > 15:
            step = max(1, num_batches // 8)
            ax.set_xticks(range(0, num_batches, step))
            ax.set_xticklabels([f"{i*self.batch_size + 1}" for i in range(0, num_batches, step)], fontsize=7)
        else:
            ax.set_xticks(range(num_batches))
            ax.set_xticklabels([f"{i*self.batch_size + 1}" for i in range(num_batches)], fontsize=7)

        ax.set_xlabel('Batch Start Episode')
        ax.set_title('Behavior Frequency Heatmap (%)')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Frequency (%)', fontsize=8)

    def _plot_consecutive_streak(self, ax: plt.Axes) -> None:
        """Chart 10: Consecutive success streak over episodes.

        Shows the running count of consecutive successes, resetting on failure.
        Includes annotation for maximum streak achieved.

        Args:
            ax: Matplotlib axes to plot on
        """
        streaks = self._compute_consecutive_streaks()

        if not streaks:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Consecutive Success Streak')
            return

        episodes = list(range(len(streaks)))
        max_streak = max(streaks)
        max_streak_idx = streaks.index(max_streak)

        # Plot streak line
        ax.fill_between(episodes, streaks, alpha=0.3, color='green')
        ax.plot(episodes, streaks, color='green', linewidth=1)

        # Reference line at target (100 consecutive)
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Target (100)')

        # Annotate max streak
        if max_streak > 0:
            ax.annotate(
                f'Max: {max_streak}',
                xy=(max_streak_idx, max_streak),
                xytext=(max_streak_idx, max_streak + max(10, max_streak * 0.1)),
                fontsize=9,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='darkgreen')
            )

        ax.set_xlabel('Episode')
        ax.set_ylabel('Consecutive Successes')
        ax.set_title('Consecutive Success Streak')
        ax.legend(loc='upper left')
        ax.set_ylim(bottom=0)

    def _plot_behavior_report_card(self, ax: plt.Axes) -> None:
        """Chart F: Behavior comparison (first 100 vs last 100 episodes).

        Horizontal bar chart comparing behavior frequencies between
        early and late training episodes.

        Args:
            ax: Matplotlib axes to plot on
        """
        num_episodes = len(self.tracker.behavior_reports)

        if num_episodes < 100:
            ax.text(0.5, 0.5, 'Need 100+ episodes', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Behavior Report Card')
            return

        # Get frequencies for first and last 100 episodes
        first_100 = self._get_behavior_frequencies(0, 100)
        last_100 = self._get_behavior_frequencies(num_episodes - 100, num_episodes)

        # Filter to report card behaviors that have data
        behaviors = []
        first_vals = []
        last_vals = []
        deltas = []

        for behavior in REPORT_CARD_BEHAVIORS:
            first_val = first_100.get(behavior, 0)
            last_val = last_100.get(behavior, 0)
            # Only include if there's meaningful data
            if first_val > 0 or last_val > 0:
                behaviors.append(behavior)
                first_vals.append(first_val)
                last_vals.append(last_val)
                deltas.append(last_val - first_val)

        if not behaviors:
            ax.text(0.5, 0.5, 'No behavior data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Behavior Report Card')
            return

        # Sort by delta (most improved at top)
        sorted_indices = np.argsort(deltas)[::-1]
        behaviors = [behaviors[i] for i in sorted_indices]
        first_vals = [first_vals[i] for i in sorted_indices]
        last_vals = [last_vals[i] for i in sorted_indices]

        # Create horizontal bar chart
        y_pos = np.arange(len(behaviors))
        bar_height = 0.35

        ax.barh(y_pos - bar_height/2, first_vals, bar_height, label='First 100', color='#3498db', alpha=0.8)
        ax.barh(y_pos + bar_height/2, last_vals, bar_height, label='Last 100', color='#e74c3c', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(behaviors, fontsize=8)
        ax.set_xlabel('Frequency (%)')
        ax.set_title('Behavior Report Card: First 100 vs Last 100')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, 105)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _compute_rolling_average(
        self,
        data: List[float],
        window: int = 50
    ) -> np.ndarray:
        """Compute rolling average of a data series.

        Args:
            data: List of values to average
            window: Window size for rolling average

        Returns:
            Numpy array of rolling averages (same length as input)
        """
        arr = np.array(data)
        if len(arr) < window:
            # Not enough data for full window, use cumulative average
            return np.cumsum(arr) / np.arange(1, len(arr) + 1)

        # Use convolution for efficient rolling average
        kernel = np.ones(window) / window
        # 'valid' mode gives us len - window + 1 points
        rolling = np.convolve(arr, kernel, mode='valid')

        # Pad the beginning with cumulative averages to match input length
        pad_size = len(arr) - len(rolling)
        cumsum_pad = np.cumsum(arr[:pad_size]) / np.arange(1, pad_size + 1)

        return np.concatenate([cumsum_pad, rolling])

    def _compute_batch_stats(
        self,
        episodes: List,
        stat_func
    ) -> Tuple[List[str], List[float]]:
        """Compute statistics for each batch of episodes.

        Args:
            episodes: List of episode data
            stat_func: Function to compute statistic for each batch

        Returns:
            Tuple of (batch_labels, batch_values)
        """
        num_episodes = len(episodes)
        num_batches = (num_episodes + self.batch_size - 1) // self.batch_size

        labels = []
        values = []

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, num_episodes)
            batch = episodes[start:end]

            if batch:
                labels.append(f"{start + 1}-{end}")
                values.append(stat_func(batch))

        return labels, values

    def _get_behavior_frequencies(
        self,
        start_idx: int,
        end_idx: int
    ) -> Dict[str, float]:
        """Get behavior frequencies for a range of episodes.

        Args:
            start_idx: Starting episode index
            end_idx: Ending episode index (exclusive)

        Returns:
            Dictionary mapping behavior names to frequency percentages
        """
        reports = self.tracker.behavior_reports[start_idx:end_idx]
        num_episodes = len(reports)

        if num_episodes == 0:
            return {}

        behavior_counts: Dict[str, int] = {}
        for report in reports:
            for behavior in report.behaviors:
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

        # Convert to percentages
        return {
            behavior: (count / num_episodes) * 100
            for behavior, count in behavior_counts.items()
        }

    def _get_outcome_distribution_per_batch(self) -> Dict[str, List[float]]:
        """Get outcome category percentages for each batch.

        Returns:
            Dictionary mapping outcome categories to lists of percentages per batch
        """
        reports = self.tracker.behavior_reports
        num_episodes = len(reports)
        num_batches = (num_episodes + self.batch_size - 1) // self.batch_size

        # Initialize result dict
        result = {cat: [] for cat in OUTCOME_CATEGORIES}

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, num_episodes)
            batch = reports[start:end]
            batch_len = len(batch)

            if batch_len == 0:
                for cat in OUTCOME_CATEGORIES:
                    result[cat].append(0.0)
                continue

            # Count outcomes in this batch
            category_counts = {cat: 0 for cat in OUTCOME_CATEGORIES}
            for report in batch:
                category = OUTCOME_TO_CATEGORY.get(report.outcome, 'crashed')
                category_counts[category] += 1

            # Convert to percentages
            for cat in OUTCOME_CATEGORIES:
                result[cat].append((category_counts[cat] / batch_len) * 100)

        return result

    def _compute_consecutive_streaks(self) -> List[int]:
        """Compute consecutive success streak at each episode.

        Returns:
            List of streak counts (same length as episode_results)
        """
        streaks = []
        current_streak = 0

        for result in self.tracker.episode_results:
            if result.success:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)

        return streaks
