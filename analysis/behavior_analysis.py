"""Behavior analysis for Lunar Lander episodes.

This module analyzes episode trajectories to detect and report behavior patterns,
enabling automated diagnosis of agent performance without manual observation.

See BEHAVIORS.md for full documentation of all detected behaviors.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# Threshold constants
class Thresholds:
    """Detection thresholds for behavior analysis."""

    # Velocity thresholds (m/s)
    CRASH_VELOCITY = 2.0
    HARD_LANDING_VELOCITY = 1.0
    SOFT_LANDING_VELOCITY = 0.5
    HOVER_VELOCITY = 0.1

    # Angle thresholds (radians)
    UPRIGHT = 0.2
    SLIGHT_TILT = 0.3
    TILTED = 0.5
    SIDEWAYS = 1.0
    FLIPPED = 1.57  # pi/2

    # Angular velocity thresholds (rad/s)
    SPINNING = 2

    # Position thresholds
    OFF_SCREEN_Y = 1.5
    OFF_SCREEN_X = 1.0
    LOW_ALTITUDE = 0.25
    MEDIUM_ALTITUDE = 0.5
    CENTERED = 0.3
    WELL_CENTERED = 0.2

    # Thrust thresholds
    THRUST_HEAVY = 0.5
    THRUST_MODERATE_LOW = 0.2
    THRUST_NONE = 0.05
    THRUST_BIAS = 0.3
    THRUST_BALANCED = 0.1

    # Variance thresholds
    LOW_VARIANCE = 0.1
    HIGH_VARIANCE = 0.5
    LOW_ACTION_VARIANCE = 0.2
    HIGH_ACTION_VARIANCE = 0.5

    # Episode length thresholds (steps)
    VERY_SHORT = 50
    SHORT = 100
    LONG = 300
    VERY_LONG = 600

    # Event detection
    SIGN_CHANGE_OSCILLATION = 6
    SIGN_CHANGE_YO_YO = 4
    SIGN_CHANGE_WOBBLE = 8
    CONSECUTIVE_FULL_THROTTLE = 10
    STABLE_STEPS = 20


@dataclass
class BehaviorReport:
    """Collection of detected behaviors for an episode.

    Attributes:
        outcome: Primary episode outcome (mutually exclusive)
        behaviors: List of all detected behavior patterns
    """

    outcome: str
    behaviors: List[str]

    def __str__(self) -> str:
        if self.behaviors:
            return f"{self.outcome} | {', '.join(self.behaviors)}"
        return self.outcome


class BehaviorAnalyzer:
    """Analyzes episode trajectories to detect behavior patterns.

    Examines observation sequences and action histories to identify
    patterns in vertical movement, horizontal drift, orientation,
    thruster usage, and landing events.
    """

    def analyze(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        terminated: bool,
        truncated: bool
    ) -> BehaviorReport:
        """Analyze an episode and return detected behaviors.

        Args:
            observations: Observation array of shape (steps, 8)
            actions: Action array of shape (steps, 2)
            terminated: Whether episode ended due to termination
            truncated: Whether episode ended due to truncation

        Returns:
            BehaviorReport containing outcome and behavior list
        """
        if len(observations) == 0:
            return BehaviorReport("UNKNOWN", [])

        n = len(observations)

        # Extract state components (single pass through observations)
        x = observations[:, 0]
        y = observations[:, 1]
        vx = observations[:, 2]
        vy = observations[:, 3]
        angle = observations[:, 4]
        angular_vel = observations[:, 5] * 2.5  # Convert to rad/s
        leg1 = observations[:, 6]
        leg2 = observations[:, 7]

        # Extract action components
        main_thrust = actions[:, 0] if len(actions) > 0 else np.array([0])
        side_thrust = actions[:, 1] if len(actions) > 0 else np.array([0])

        # Pre-compute common statistics once (avoids redundant passes)
        stats = {
            # Means
            'mean_vy': np.mean(vy),
            'mean_vx': np.mean(vx),
            'mean_angle': np.mean(angle),
            'mean_main': np.mean(main_thrust),
            'mean_side': np.mean(side_thrust),
            'mean_abs_angular_vel': np.mean(np.abs(angular_vel)),
            # Variances
            'var_y': np.var(y),
            'var_vy': np.var(vy),
            'var_x': np.var(x),
            # Min/Max
            'min_y': np.min(y),
            'max_y': np.max(y),
            'max_abs_angle': np.max(np.abs(angle)),
            'max_abs_vx': np.max(np.abs(vx)),
            'max_abs_angular_vel': np.max(np.abs(angular_vel)),
            # Sign changes (computed once)
            'vy_sign_changes': np.sum(np.diff(np.sign(vy)) != 0),
            'vx_sign_changes': np.sum(np.diff(np.sign(vx)) != 0),
            'angle_sign_changes': np.sum(np.diff(np.sign(angle)) != 0),
            # Finals
            'final_x': x[-1],
            'final_y': y[-1],
            'final_vy': vy[-1],
            'final_vx': vx[-1],
            'final_angle': angle[-1],
            'final_angular_vel': angular_vel[-1],
            'final_leg1': leg1[-1],
            'final_leg2': leg2[-1],
            # Initials
            'initial_x': x[0],
            'initial_y': y[0],
            # Counts
            'n': n,
        }

        # Detect outcome first
        outcome = self._detect_outcome(
            x, y, vx, vy, angle, angular_vel, leg1, leg2,
            terminated, truncated, stats
        )

        # Collect all behaviors
        behaviors: List[str] = []

        behaviors.extend(self._detect_vertical_patterns(y, vy, main_thrust, stats))
        behaviors.extend(self._detect_horizontal_patterns(x, vx, stats))
        behaviors.extend(self._detect_orientation_patterns(angle, angular_vel, stats))
        behaviors.extend(self._detect_thruster_patterns(main_thrust, side_thrust, stats))
        behaviors.extend(self._detect_contact_events(leg1, leg2))
        behaviors.extend(self._detect_efficiency(n))
        behaviors.extend(self._detect_trajectory_patterns(x, y, vy, angle, angular_vel, main_thrust, stats))
        behaviors.extend(self._detect_altitude_milestones(y, stats))
        behaviors.extend(self._detect_critical_moments(angle, angular_vel, outcome, stats))

        # Detect hovering behaviors until timeout
        if truncated:
            low_altitude_pct = np.mean(y < Thresholds.LOW_ALTITUDE)
            centered_pct = np.mean(np.abs(x) < Thresholds.CENTERED)
            low_velocity = np.mean(np.abs(vy)) < Thresholds.HOVER_VELOCITY

            # Hovering near ground (regardless of horizontal position)
            if low_altitude_pct > 0.5 and low_velocity:
                behaviors.append("HOVER_NEAR_GROUND_TIMEOUT")

            # Hovering over goal specifically (centered + low altitude)
            if low_altitude_pct > 0.5 and centered_pct > 0.5 and low_velocity:
                behaviors.append("HOVERED_OVER_GOAL_TIMEOUT")

        return BehaviorReport(outcome, behaviors)

    def _detect_outcome(
        self,
        x: np.ndarray,
        y: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        angle: np.ndarray,
        angular_vel: np.ndarray,
        leg1: np.ndarray,
        leg2: np.ndarray,
        terminated: bool,
        truncated: bool,
        stats: dict
    ) -> str:
        """Determine the primary episode outcome."""
        final_x = stats['final_x']
        final_y = stats['final_y']
        final_vy = stats['final_vy']
        final_vx = stats['final_vx']
        final_angle = stats['final_angle']
        final_angular_vel = stats['final_angular_vel']
        final_leg1 = stats['final_leg1']
        final_leg2 = stats['final_leg2']

        final_velocity = np.sqrt(final_vx**2 + final_vy**2)
        both_legs = final_leg1 > 0.5 and final_leg2 > 0.5
        one_leg = (final_leg1 > 0.5) != (final_leg2 > 0.5)

        # Check for flying off screen
        # Consider both position and velocity/tilt to determine direction
        off_top = final_y > Thresholds.OFF_SCREEN_Y
        off_left = final_x < -Thresholds.OFF_SCREEN_X
        off_right = final_x > Thresholds.OFF_SCREEN_X

        if off_top or off_left or off_right:
            # Determine primary direction based on velocity and tilt
            # If significantly tilted and moving sideways, prioritize sideways classification
            is_tilted = abs(final_angle) > Thresholds.TILTED
            moving_left = final_vx < -0.3
            moving_right = final_vx > 0.3
            moving_up = final_vy > 0.3

            # Prioritize sideways if tilted and moving sideways
            if is_tilted and (moving_left or off_left) and not moving_up:
                return "FLEW_OFF_LEFT_TILTED"
            if is_tilted and (moving_right or off_right) and not moving_up:
                return "FLEW_OFF_RIGHT_TILTED"

            # Standard off-screen classifications
            if off_left:
                return "FLEW_OFF_LEFT"
            if off_right:
                return "FLEW_OFF_RIGHT"
            if off_top:
                return "FLEW_OFF_TOP"

        # Check for successful landings (both legs on ground)
        if both_legs:
            if (final_velocity < Thresholds.SOFT_LANDING_VELOCITY and
                    abs(final_angle) < Thresholds.UPRIGHT and
                    abs(final_x) < Thresholds.WELL_CENTERED):
                return "LANDED_PERFECTLY"
            if (final_velocity < Thresholds.HARD_LANDING_VELOCITY and
                    abs(final_angle) < Thresholds.SLIGHT_TILT):
                return "LANDED_SOFTLY"
            if abs(final_angle) > Thresholds.SLIGHT_TILT:
                return "LANDED_TILTED"
            if final_velocity < Thresholds.CRASH_VELOCITY:
                return "LANDED_HARD"
            # High horizontal velocity with both legs down = sliding
            if abs(final_vx) > 0.1:
                return "LANDED_SLIDING"

        # Check for single leg landing
        if one_leg and terminated:
            return "LANDED_ONE_LEG"

        # Check for truncation outcomes (timeout)
        if truncated:
            # Timed out with both legs on ground = successful landing that ran out of time
            if both_legs:
                return "TIMED_OUT_ON_GROUND"
            y_variance = np.var(y[-50:]) if stats['n'] > 50 else stats['var_y']
            if y_variance < Thresholds.LOW_VARIANCE and abs(final_vy) < Thresholds.HOVER_VELOCITY:
                return "TIMED_OUT_HOVERING"
            if final_vy > Thresholds.HOVER_VELOCITY:
                return "TIMED_OUT_ASCENDING"
            return "TIMED_OUT_DESCENDING"

        # Check for crashes (terminated without good landing)
        if terminated:
            if abs(final_angular_vel) > Thresholds.SPINNING:
                return "CRASHED_SPINNING"
            if abs(final_angle) > Thresholds.SIDEWAYS:
                return "CRASHED_SIDEWAYS"
            if abs(final_angle) > Thresholds.TILTED:
                return "CRASHED_FAST_TILTED"
            # Fast vertical crash: roughly upright but high downward velocity
            if abs(final_angle) <= Thresholds.TILTED and abs(final_vy) > Thresholds.HARD_LANDING_VELOCITY:
                return "CRASHED_FAST_VERTICAL"
            # Catch-all for other crash scenarios
            return "CRASHED_OTHER"

        return "UNKNOWN"

    def _detect_vertical_patterns(
        self,
        y: np.ndarray,
        vy: np.ndarray,
        main_thrust: np.ndarray,
        stats: dict
    ) -> List[str]:
        """Detect vertical flight patterns."""
        behaviors = []
        mean_vy = stats['mean_vy']
        vy_variance = stats['var_vy']
        y_variance = stats['var_y']

        # Descent patterns
        if -0.5 < mean_vy < -0.1 and vy_variance < Thresholds.HIGH_VARIANCE:
            behaviors.append("CONTROLLED_DESCENT")
        elif -0.1 <= mean_vy < 0 and np.sum(vy < 0) > len(vy) * 0.5:
            behaviors.append("SLOW_DESCENT")
        elif mean_vy < -0.5:
            behaviors.append("RAPID_DESCENT")

        # Hover detection
        if y_variance < Thresholds.LOW_VARIANCE:
            behaviors.append("HOVER_MAINTAINED")

        # Ascent detection
        if len(y) > 1 and y[-1] - y[0] > 0.3:
            behaviors.append("ASCENDED")

        # Yo-yo pattern (velocity sign changes)
        if stats['vy_sign_changes'] > Thresholds.SIGN_CHANGE_YO_YO:
            behaviors.append("YO_YO_PATTERN")

        # Continuous burn
        if np.mean(main_thrust > 0.3) > 0.8:
            behaviors.append("CONTINUOUS_BURN")

        # Late braking detection
        if len(main_thrust) > 20:
            early_thrust = np.mean(main_thrust[:int(len(main_thrust) * 0.8)])
            late_thrust = np.mean(main_thrust[int(len(main_thrust) * 0.8):])
            if early_thrust < 0.2 and late_thrust > 0.5:
                behaviors.append("LATE_BRAKING")

        # Freefall detection (low thrust, velocity matching gravity)
        if np.mean(main_thrust) < 0.1 and mean_vy < -0.3:
            behaviors.append("FREEFALL")

        # Stalled then fell
        if len(y) > 50:
            first_portion = y[:int(len(y) * 0.3)]
            if np.var(first_portion) < Thresholds.LOW_VARIANCE:
                later_vy = vy[int(len(vy) * 0.3):]
                if np.mean(later_vy) < -0.4:
                    behaviors.append("STALLED_THEN_FELL")

        return behaviors

    def _detect_horizontal_patterns(
        self,
        x: np.ndarray,
        vx: np.ndarray,
        stats: dict
    ) -> List[str]:
        """Detect horizontal movement patterns."""
        behaviors = []
        final_x = stats['final_x']
        initial_x = stats['initial_x']

        # Stayed centered
        if np.all(np.abs(x) < Thresholds.CENTERED):
            behaviors.append("STAYED_CENTERED")
        elif final_x < -Thresholds.CENTERED and abs(initial_x) < Thresholds.CENTERED:
            behaviors.append("DRIFTED_LEFT")
        elif final_x > Thresholds.CENTERED and abs(initial_x) < Thresholds.CENTERED:
            behaviors.append("DRIFTED_RIGHT")

        # Returned to center
        if (np.max(np.abs(x)) > Thresholds.CENTERED and
                abs(final_x) < Thresholds.WELL_CENTERED):
            behaviors.append("RETURNED_TO_CENTER")

        # Horizontal oscillation
        if stats['vx_sign_changes'] > Thresholds.SIGN_CHANGE_OSCILLATION:
            behaviors.append("HORIZONTAL_OSCILLATION")

        # Strong lateral velocity
        if stats['max_abs_vx'] > 1.0:
            behaviors.append("STRONG_LATERAL_VELOCITY")

        return behaviors

    def _detect_orientation_patterns(
        self,
        angle: np.ndarray,
        angular_vel: np.ndarray,
        stats: dict
    ) -> List[str]:
        """Detect orientation and stability patterns."""
        behaviors = []
        mean_angle = stats['mean_angle']
        final_angle = stats['final_angle']
        max_angle = stats['max_abs_angle']

        # Upright throughout
        if np.all(np.abs(angle) < Thresholds.UPRIGHT):
            behaviors.append("STAYED_UPRIGHT")

        # Lean patterns
        if -0.5 < mean_angle < -0.1:
            behaviors.append("SLIGHT_LEFT_LEAN")
        elif 0.1 < mean_angle < 0.5:
            behaviors.append("SLIGHT_RIGHT_LEAN")

        # Heavy tilt
        if np.min(angle) < -Thresholds.TILTED:
            behaviors.append("HEAVY_LEFT_TILT")
        if np.max(angle) > Thresholds.TILTED:
            behaviors.append("HEAVY_RIGHT_TILT")

        # Flipped over
        if max_angle > Thresholds.FLIPPED:
            behaviors.append("FLIPPED_OVER")

        # Spinning
        if stats['max_abs_angular_vel'] > Thresholds.SPINNING:
            behaviors.append("SPINNING_UNCONTROLLED")

        # Recovered from tilt
        if max_angle > Thresholds.TILTED and abs(final_angle) < Thresholds.UPRIGHT:
            behaviors.append("RECOVERED_FROM_TILT")

        # Progressive tilt (angle magnitude increasing)
        n = stats['n']
        if n > 20:
            early_tilt = np.mean(np.abs(angle[:int(n * 0.3)]))
            late_tilt = np.mean(np.abs(angle[int(n * 0.7):]))
            if late_tilt > early_tilt + 0.2:
                behaviors.append("PROGRESSIVE_TILT")

        # Wobbling
        if stats['angle_sign_changes'] > Thresholds.SIGN_CHANGE_WOBBLE:
            behaviors.append("WOBBLING")

        return behaviors

    def _detect_thruster_patterns(
        self,
        main_thrust: np.ndarray,
        side_thrust: np.ndarray,
        stats: dict
    ) -> List[str]:
        """Detect thruster usage patterns."""
        behaviors = []
        mean_main = stats['mean_main']
        mean_side = stats['mean_side']
        action_std = np.std(np.concatenate([main_thrust, side_thrust]))

        # Main thrust levels
        if mean_main > Thresholds.THRUST_HEAVY:
            behaviors.append("MAIN_THRUST_HEAVY")
        elif mean_main > Thresholds.THRUST_MODERATE_LOW:
            behaviors.append("MAIN_THRUST_MODERATE")
        elif mean_main > Thresholds.THRUST_NONE:
            behaviors.append("MAIN_THRUST_LIGHT")
        else:
            behaviors.append("MAIN_THRUST_NONE")

        # Side thrust bias
        if mean_side < -Thresholds.THRUST_BIAS:
            behaviors.append("SIDE_THRUST_LEFT_BIAS")
        elif mean_side > Thresholds.THRUST_BIAS:
            behaviors.append("SIDE_THRUST_RIGHT_BIAS")
        elif abs(mean_side) < Thresholds.THRUST_BALANCED:
            behaviors.append("SIDE_THRUST_BALANCED")

        # Thrust variance
        if action_std > Thresholds.HIGH_ACTION_VARIANCE:
            behaviors.append("ERRATIC_THRUST")
        elif action_std < Thresholds.LOW_ACTION_VARIANCE:
            behaviors.append("SMOOTH_THRUST")

        # Full throttle bursts
        consecutive_full = 0
        max_consecutive = 0
        for t in main_thrust:
            if t > 0.95:
                consecutive_full += 1
                max_consecutive = max(max_consecutive, consecutive_full)
            else:
                consecutive_full = 0

        if max_consecutive >= Thresholds.CONSECUTIVE_FULL_THROTTLE:
            behaviors.append("FULL_THROTTLE_BURST")

        return behaviors

    def _detect_contact_events(
        self,
        leg1: np.ndarray,
        leg2: np.ndarray
    ) -> List[str]:
        """Detect leg contact events."""
        behaviors = []

        had_leg1 = np.any(leg1 > 0.5)
        had_leg2 = np.any(leg2 > 0.5)
        had_both = np.any((leg1 > 0.5) & (leg2 > 0.5))

        # No contact
        if not had_leg1 and not had_leg2:
            behaviors.append("NO_CONTACT_MADE")
            return behaviors

        # Single leg scrapes
        if had_leg1 and not had_leg2:
            behaviors.append("SCRAPED_LEFT_LEG")
        elif had_leg2 and not had_leg1:
            behaviors.append("SCRAPED_RIGHT_LEG")

        # Clean touchdown (both legs within 5 steps)
        if had_both:
            leg1_first = np.argmax(leg1 > 0.5) if had_leg1 else len(leg1)
            leg2_first = np.argmax(leg2 > 0.5) if had_leg2 else len(leg2)
            if abs(leg1_first - leg2_first) <= 5:
                behaviors.append("TOUCHED_DOWN_CLEAN")

        # Bounced (had contact, lost it)
        any_contact = (leg1 > 0.5) | (leg2 > 0.5)
        if np.any(any_contact):
            contact_indices = np.where(any_contact)[0]
            if len(contact_indices) > 0:
                first_contact = contact_indices[0]
                after_contact = any_contact[first_contact:]
                if np.any(~after_contact[1:]):  # Lost contact after first touch
                    # Check if contact was re-established
                    lost_then_regained = False
                    in_contact = True
                    for c in after_contact:
                        if in_contact and not c:
                            in_contact = False
                        elif not in_contact and c:
                            lost_then_regained = True
                            break
                    if lost_then_regained:
                        behaviors.append("BOUNCED")
                        behaviors.append("MULTIPLE_TOUCHDOWNS")

        # Prolonged one leg contact
        leg1_only = (leg1 > 0.5) & (leg2 < 0.5)
        leg2_only = (leg2 > 0.5) & (leg1 < 0.5)
        if np.sum(leg1_only) > 20 or np.sum(leg2_only) > 20:
            behaviors.append("PROLONGED_ONE_LEG")

        return behaviors

    def _detect_efficiency(self, num_steps: int) -> List[str]:
        """Classify episode duration."""
        behaviors = []

        if num_steps < Thresholds.VERY_SHORT:
            behaviors.append("VERY_SHORT_EPISODE")
        elif num_steps < Thresholds.SHORT:
            behaviors.append("SHORT_EPISODE")
        elif num_steps < Thresholds.LONG:
            behaviors.append("STANDARD_EPISODE")
        elif num_steps < Thresholds.VERY_LONG:
            behaviors.append("LONG_EPISODE")
        else:
            behaviors.append("VERY_LONG_EPISODE")

        return behaviors

    def _detect_trajectory_patterns(
        self,
        x: np.ndarray,
        y: np.ndarray,
        vy: np.ndarray,
        angle: np.ndarray,
        angular_vel: np.ndarray,
        main_thrust: np.ndarray,
        stats: dict
    ) -> List[str]:
        """Detect overall trajectory patterns."""
        behaviors = []

        x_variance = stats['var_x']
        descending = stats['mean_vy'] < -0.05

        # Direct approach
        if x_variance < Thresholds.LOW_VARIANCE and descending:
            behaviors.append("DIRECT_APPROACH")
        elif x_variance > Thresholds.HIGH_VARIANCE and descending:
            behaviors.append("CURVED_APPROACH")

        # Spiral descent
        if descending and stats['mean_abs_angular_vel'] > 0.3:
            behaviors.append("SPIRAL_DESCENT")

        # Zigzag descent
        if descending:
            x_sign_changes = np.sum(np.diff(np.sign(x - np.mean(x))) != 0)
            if x_sign_changes > 4:
                behaviors.append("ZIGZAG_DESCENT")

        # Suicide burn
        if len(vy) > 20 and len(main_thrust) > 20:
            split_point = int(len(vy) * 0.85)
            early_velocity = np.mean(np.abs(vy[:split_point]))
            late_thrust = np.mean(main_thrust[split_point:])
            if early_velocity > 0.5 and late_thrust > 0.6:
                behaviors.append("SUICIDE_BURN")

        # Gradual slowdown
        if len(vy) > 20:
            # Check if |vy| decreased over episode
            vy_abs = np.abs(vy)
            thirds = np.array_split(vy_abs, 3)
            if len(thirds) == 3:
                means = [np.mean(t) for t in thirds]
                if means[0] > means[1] > means[2]:
                    behaviors.append("GRADUAL_SLOWDOWN")

        return behaviors

    def _detect_altitude_milestones(self, y: np.ndarray, stats: dict) -> List[str]:
        """Detect altitude-related events."""
        behaviors = []
        initial_y = stats['initial_y']
        min_y = stats['min_y']
        max_y = stats['max_y']

        if min_y < Thresholds.LOW_ALTITUDE:
            behaviors.append("REACHED_LOW_ALTITUDE")

        if min_y > Thresholds.MEDIUM_ALTITUDE:
            behaviors.append("STAYED_HIGH")

        if max_y > initial_y + 0.1:
            behaviors.append("PEAKED_ABOVE_START")

        # Ground approach abort
        if min_y < 0.3:
            min_idx = np.argmin(y)
            if min_idx < len(y) - 10:
                after_min = y[min_idx:]
                if np.max(after_min) - min_y > 0.2:
                    behaviors.append("GROUND_APPROACH_ABORT")

        return behaviors

    def _detect_critical_moments(
        self,
        angle: np.ndarray,
        angular_vel: np.ndarray,
        outcome: str,
        stats: dict
    ) -> List[str]:
        """Detect phase-based critical behavior patterns."""
        behaviors = []
        n = stats['n']

        if n < 20:
            return behaviors

        # Never stabilized
        stable_window = 0
        for i in range(n):
            if abs(angle[i]) < Thresholds.SLIGHT_TILT:
                stable_window += 1
                if stable_window >= Thresholds.STABLE_STEPS:
                    break
            else:
                stable_window = 0
        else:
            behaviors.append("NEVER_STABILIZED")

        # Controlled throughout
        if (np.all(np.abs(angle) < 0.4) and
                np.all(np.abs(angular_vel) < 0.5)):
            behaviors.append("CONTROLLED_THROUGHOUT")

        # Lost control late
        if n > 50:
            early_angular = np.mean(np.abs(angular_vel[:int(n * 0.7)]))
            late_angular = np.mean(np.abs(angular_vel[int(n * 0.7):]))
            if early_angular < 0.3 and late_angular > 0.8:
                behaviors.append("LOST_CONTROL_LATE")

        # Overcorrected to crash (increasing oscillation amplitude)
        if "CRASHED" in outcome and n > 30:
            first_half_var = np.var(angle[:n // 2])
            second_half_var = np.var(angle[n // 2:])
            if second_half_var > first_half_var * 2:
                behaviors.append("OVERCORRECTED_TO_CRASH")

        return behaviors
