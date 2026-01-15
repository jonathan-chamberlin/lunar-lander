"""Training visualization charts for Lunar Lander RL.

This module generates charts to visualize training progress and effectiveness.
Single Responsibility: Transform collected metrics into visual charts.
"""

from typing import List, Dict, Tuple, Optional
import logging

import matplotlib.pyplot as plt
import numpy as np

from analysis.diagnostics import DiagnosticsTracker
from constants import (
    REPORT_CARD_BEHAVIORS,
    OUTCOME_CATEGORY_ORDER,
    OUTCOME_COLORS,
    OUTCOME_TO_CATEGORY
)

logger = logging.getLogger(__name__)


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
    # Main Public Methods
    # =========================================================================

    def generate_to_file(self, file_path: str) -> bool:
        """Generate charts to a file without GUI.

        Uses 'Agg' backend to avoid GUI conflicts with multiprocessing.
        This is safe to call while AsyncVectorEnv is running.

        Args:
            file_path: Path to save the chart image

        Returns:
            True if successful, False otherwise
        """
        import matplotlib
        original_backend = matplotlib.get_backend()

        try:
            # Switch to non-GUI backend
            plt.switch_backend('Agg')

            # Generate the figure (don't show)
            fig = self._create_figure()
            if fig is None:
                return False

            # Save to file
            fig.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return True

        except Exception as e:
            logger.warning(f"Failed to generate chart to file: {e}")
            return False
        finally:
            # Restore original backend
            try:
                plt.switch_backend(original_backend)
            except Exception:
                pass  # Backend switch may fail if original was interactive

    def generate_all(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
        block: bool = True
    ) -> Optional[plt.Figure]:
        """Generate all charts in a single figure with subplots.

        Creates a 2x4 grid of subplots containing all training visualizations.

        Args:
            show: Whether to display the figure with plt.show()
            save_path: Optional path to save the figure as an image
            block: Whether plt.show() blocks execution (False for periodic updates)

        Returns:
            The matplotlib Figure object, or None if not enough data
        """
        fig = self._create_figure()
        if fig is None:
            return None

        # Set window size to 95% of screen width
        try:
            fig_manager = plt.get_current_fig_manager()
            if hasattr(fig_manager, 'window'):
                # Get screen dimensions using tkinter
                window = fig_manager.window
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()

                # Calculate 95% width and proportional height (maintain aspect ratio)
                target_width = int(screen_width * 0.95)
                # Original figure is 12.5x6.5, so height = width * (6.5/12.5)
                target_height = int(target_width * (6.5 / 12.5))

                # Center the window horizontally
                x_position = int((screen_width - target_width) / 2)
                y_position = 50  # Small offset from top

                # Set geometry: widthxheight+x+y
                window.geometry(f"{target_width}x{target_height}+{x_position}+{y_position}")
                window.minsize(950, 650)
        except Exception:
            pass  # Not all backends support this

        # Save if path provided
        if save_path:
            try:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Charts saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save charts: {e}")

        # Show if requested
        if show:
            plt.show(block=block)
        else:
            plt.close(fig)
            return None

        return fig

    def _create_figure(self) -> Optional[plt.Figure]:
        """Create the chart figure with all subplots.

        Returns:
            The matplotlib Figure object, or None if not enough data
        """
        # Check if we have enough data
        num_episodes = len(self.tracker.episode_results)
        if num_episodes < 10:
            logger.warning(f"Not enough episodes ({num_episodes}) to generate meaningful charts")
            return None

        # Create figure with 2x4 subplot grid
        fig, axes = plt.subplots(
            2, 4,
            figsize=(12.5, 6.5),
            constrained_layout=True
        )
        fig.suptitle(f'Training Progress - {num_episodes} Episodes', fontsize=12, fontweight='bold')

        # Chart configurations: (axes_position, plot_method, chart_name)
        chart_configs = [
            (axes[0, 0], self._plot_reward_over_time, "reward"),
            (axes[0, 1], self._plot_success_rate, "success rate"),
            (axes[0, 2], self._plot_outcome_stacked_area, "outcome"),
            (axes[0, 3], self._plot_episode_duration, "duration"),
            (axes[1, 0], self._plot_consecutive_streak, "streak"),
            (axes[1, 1], self._plot_behavior_heatmap, "behavior heatmap"),
            (axes[1, 2], self._plot_behavior_report_card, "report card"),
            (axes[1, 3], self._plot_landing_reward_histogram, "landing histogram"),
        ]

        # Generate each chart with error handling
        for ax, plot_method, name in chart_configs:
            self._safe_plot(ax, plot_method, name)

        return fig

    def _safe_plot(self, ax: plt.Axes, plot_method, name: str) -> None:
        """Safely execute a plot method with error handling.

        Args:
            ax: Matplotlib axes to plot on
            plot_method: Method to call for plotting
            name: Human-readable name for error messages
        """
        try:
            plot_method(ax)
        except Exception as e:
            logger.warning(f"Failed to plot {name} chart: {e}")
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)

    # =========================================================================
    # Individual Chart Methods
    # =========================================================================

    def _plot_reward_over_time(self, ax: plt.Axes) -> None:
        """Chart 1: Episode reward as stacked bars (env + shaped).

        Shows env_reward (bottom) and shaped_bonus (top) as stacked bars.
        The total height shows total_reward. Includes success threshold line.

        Args:
            ax: Matplotlib axes to plot on
        """
        results = self.tracker.episode_results

        if not results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Episode Reward Over Time')
            return

        episodes = np.arange(len(results))
        env_rewards = np.array([r.env_reward for r in results])
        shaped_bonuses = np.array([r.shaped_bonus for r in results])

        # Stacked bars: env_reward on bottom, shaped_bonus on top
        # Use width=1 for edge-to-edge bars (appears continuous at scale)
        ax.bar(episodes, env_rewards, width=1.0, color='#3498db', alpha=0.8, label='Env Reward')
        ax.bar(episodes, shaped_bonuses, width=1.0, bottom=env_rewards, color='#e74c3c', alpha=0.8, label='Shaped Bonus')

        # Rolling average line for total reward
        total_rewards = env_rewards + shaped_bonuses
        rolling_avg = self._compute_rolling_average(total_rewards.tolist(), window=50)
        ax.plot(episodes, rolling_avg, color='yellow', linewidth=1.5, label='50-Ep Avg (Total)')

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.9, label='Env Reward Success (200)')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward Over Time (Stacked)')
        ax.legend(loc='lower right', fontsize=7)

    def _plot_episode_duration(self, ax: plt.Axes) -> None:
        """Chart: Episode duration over time.

        Shows episode duration (seconds) with rolling average.

        Args:
            ax: Matplotlib axes to plot on
        """
        results = self.tracker.episode_results

        if not results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Episode Duration')
            return

        episodes = np.arange(len(results))
        durations = np.array([r.duration_seconds for r in results])

        # Plot raw durations as transparent scatter
        ax.scatter(episodes, durations, alpha=0.3, s=8, color='purple', label='Duration')

        # Rolling average line
        rolling_avg = self._compute_rolling_average(durations.tolist(), window=50)
        ax.plot(episodes, rolling_avg, color='purple', linewidth=2, label='50-Ep Average')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Duration (seconds)')
        ax.set_title('Episode Duration')
        ax.legend(loc='upper right', fontsize=7)
        ax.set_ylim(bottom=0)

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
            ax.set_title('Outcome Distribution')
            return

        outcome_data = self._get_outcome_distribution_per_batch()
        num_batches = len(outcome_data['landed'])

        if num_batches == 0:
            ax.text(0.5, 0.5, 'No batch data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Outcome Distribution')
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
        ax.set_title('Outcome Distribution')
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
            ax.set_title('Report Card')
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
            ax.set_title('Report Card')
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
        ax.set_title('Report Card (First vs Last 100)')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, 105)

    def _plot_landing_reward_histogram(self, ax: plt.Axes) -> None:
        """Chart: Histogram of env_reward for landed episodes.

        Shows distribution of env_rewards when the agent lands,
        with the 200 success threshold marked.

        Args:
            ax: Matplotlib axes to plot on
        """
        # Get env_reward distribution from tracker
        dist = self.tracker.get_env_reward_distribution()

        if 'landed' not in dist or not dist['landed']['rewards']:
            ax.text(0.5, 0.5, 'No landing data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Env Reward Distribution')
            return

        rewards = dist['landed']['rewards']

        # Create histogram with fixed bins spanning typical reward range
        bins = np.arange(-200, 320, 20)  # -200 to 300 in steps of 20
        counts, edges, patches = ax.hist(rewards, bins=bins, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)

        # Color bars based on whether they're above/below 200
        for i, (patch, edge) in enumerate(zip(patches, edges[:-1])):
            if edge >= 200:
                patch.set_facecolor('#2ecc71')  # Green for success
            elif edge >= 180:
                patch.set_facecolor('#f39c12')  # Orange for near-miss
            else:
                patch.set_facecolor('#3498db')  # Blue for below

        # Add success threshold line
        ax.axvline(x=200, color='green', linestyle='--', linewidth=2, label='Success (200)')

        # Add statistics annotation
        mean_reward = np.mean(rewards)
        above_200 = sum(1 for r in rewards if r >= 200)
        pct_above = above_200 / len(rewards) * 100

        ax.annotate(f'Mean: {mean_reward:.0f}\n>= 200: {pct_above:.1f}%',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Env Reward')
        ax.set_ylabel('Count')
        ax.set_title('Env Reward Distribution')
        ax.legend(loc='upper left', fontsize=7)

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
        result = {cat: [] for cat in OUTCOME_CATEGORY_ORDER}

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, num_episodes)
            batch = reports[start:end]
            batch_len = len(batch)

            if batch_len == 0:
                for cat in OUTCOME_CATEGORY_ORDER:
                    result[cat].append(0.0)
                continue

            # Count outcomes in this batch
            category_counts = {cat: 0 for cat in OUTCOME_CATEGORY_ORDER}
            for report in batch:
                category = OUTCOME_TO_CATEGORY.get(report.outcome, 'crashed')
                category_counts[category] += 1

            # Convert to percentages
            for cat in OUTCOME_CATEGORY_ORDER:
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
