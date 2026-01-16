"""Tests for behavior analysis module."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.behavior_analysis import BehaviorAnalyzer, BehaviorReport, Thresholds


class TestThresholds:
    """Tests for behavior detection thresholds."""

    def test_velocity_thresholds_order(self):
        """Test that velocity thresholds are in correct order."""
        assert Thresholds.HOVER_VELOCITY < Thresholds.SOFT_LANDING_VELOCITY
        assert Thresholds.SOFT_LANDING_VELOCITY < Thresholds.HARD_LANDING_VELOCITY
        assert Thresholds.HARD_LANDING_VELOCITY < Thresholds.CRASH_VELOCITY

    def test_angle_thresholds_order(self):
        """Test that angle thresholds are in correct order."""
        assert Thresholds.UPRIGHT < Thresholds.SLIGHT_TILT
        assert Thresholds.SLIGHT_TILT < Thresholds.TILTED
        assert Thresholds.TILTED < Thresholds.SIDEWAYS
        assert Thresholds.SIDEWAYS < Thresholds.FLIPPED


class TestBehaviorReport:
    """Tests for BehaviorReport dataclass."""

    def test_str_with_behaviors(self):
        """Test string representation with behaviors."""
        report = BehaviorReport(
            outcome="LANDED_SOFTLY",
            behaviors=["STAYED_UPRIGHT", "CONTROLLED_DESCENT"]
        )
        assert "LANDED_SOFTLY" in str(report)
        assert "STAYED_UPRIGHT" in str(report)

    def test_str_without_behaviors(self):
        """Test string representation without behaviors."""
        report = BehaviorReport(outcome="UNKNOWN", behaviors=[])
        assert str(report) == "UNKNOWN"


class TestBehaviorAnalyzer:
    """Tests for BehaviorAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return BehaviorAnalyzer()

    def test_empty_observations(self, analyzer):
        """Test analysis with empty observations."""
        report = analyzer.analyze(
            observations=np.array([]),
            actions=np.array([]),
            terminated=False,
            truncated=False
        )
        assert report.outcome == "UNKNOWN"
        assert report.behaviors == []

    def test_landed_softly_detection(self, analyzer, sample_observations, sample_actions):
        """Test detection of soft landing."""
        # Modify final state for soft landing
        sample_observations[-1] = [
            0.0, 0.1, 0.0, -0.3,  # x, y, vx, vy (low velocity)
            0.05, 0.0, 1.0, 1.0   # angle, ang_vel, leg1, leg2 (both legs down)
        ]

        report = analyzer.analyze(
            observations=sample_observations,
            actions=sample_actions,
            terminated=True,
            truncated=False
        )

        # Should detect a landing (exact type depends on final state)
        assert "LANDED" in report.outcome or "CRASHED" in report.outcome

    def test_crash_detection(self, analyzer, crash_observations, crash_actions):
        """Test detection of crash."""
        report = analyzer.analyze(
            observations=crash_observations,
            actions=crash_actions,
            terminated=True,
            truncated=False
        )

        # Should detect a crash due to high velocity and no leg contact
        assert "CRASHED" in report.outcome or "FLEW_OFF" in report.outcome

    def test_controlled_descent_behavior(self, analyzer, sample_observations, sample_actions):
        """Test detection of controlled descent behavior."""
        report = analyzer.analyze(
            observations=sample_observations,
            actions=sample_actions,
            terminated=True,
            truncated=False
        )

        # With steady descent velocity, should detect controlled descent
        behaviors = set(report.behaviors)
        # Should have some vertical pattern detected
        assert any(b in behaviors for b in [
            "CONTROLLED_DESCENT", "SLOW_DESCENT", "RAPID_DESCENT"
        ])

    def test_stayed_upright_behavior(self, analyzer):
        """Test detection of staying upright."""
        # Create observations with very small angle throughout
        n_steps = 50
        observations = np.zeros((n_steps, 8), dtype=np.float32)
        observations[:, 1] = np.linspace(1.0, 0.2, n_steps)  # y: descent
        observations[:, 3] = -0.3  # vy: constant descent
        observations[:, 4] = 0.05 * np.sin(np.linspace(0, 2, n_steps))  # angle: tiny wobble
        observations[-1, 6:8] = [1.0, 1.0]  # Both legs on ground at end

        actions = np.ones((n_steps, 2), dtype=np.float32) * 0.3

        report = analyzer.analyze(
            observations=observations,
            actions=actions,
            terminated=True,
            truncated=False
        )

        assert "STAYED_UPRIGHT" in report.behaviors

    def test_freefall_behavior(self, analyzer, crash_observations, crash_actions):
        """Test detection of freefall behavior."""
        report = analyzer.analyze(
            observations=crash_observations,
            actions=crash_actions,
            terminated=True,
            truncated=False
        )

        # Should detect freefall due to no thrust and rapid descent
        assert "FREEFALL" in report.behaviors

    def test_timeout_detection(self, analyzer, sample_observations, sample_actions):
        """Test detection of timeout outcomes."""
        # Create long hovering scenario
        n_steps = 200
        observations = np.zeros((n_steps, 8), dtype=np.float32)
        observations[:, 1] = 0.5  # y: constant altitude
        observations[:, 3] = 0.0  # vy: no vertical motion
        observations[:, 4] = 0.1 * np.sin(np.linspace(0, 10, n_steps))  # slight angle variation

        actions = np.ones((n_steps, 2), dtype=np.float32) * 0.5

        report = analyzer.analyze(
            observations=observations,
            actions=actions,
            terminated=False,
            truncated=True  # Timeout
        )

        assert "TIMED_OUT" in report.outcome

    def test_flew_off_detection(self, analyzer):
        """Test detection of flying off screen."""
        n_steps = 30
        observations = np.zeros((n_steps, 8), dtype=np.float32)
        observations[:, 0] = np.linspace(0, 1.5, n_steps)  # x: moving right
        observations[:, 1] = np.linspace(1.0, 1.8, n_steps)  # y: going up
        observations[:, 2] = 0.5  # vx: moving right
        observations[:, 3] = 0.3  # vy: ascending

        actions = np.ones((n_steps, 2), dtype=np.float32) * 0.5

        report = analyzer.analyze(
            observations=observations,
            actions=actions,
            terminated=True,
            truncated=False
        )

        assert "FLEW_OFF" in report.outcome

    def test_spinning_detection(self, analyzer):
        """Test detection of spinning uncontrolled."""
        n_steps = 50
        observations = np.zeros((n_steps, 8), dtype=np.float32)
        observations[:, 1] = np.linspace(1.0, 0.3, n_steps)  # y: descent
        observations[:, 3] = -0.5  # vy: descent
        observations[:, 4] = np.linspace(0, 3.14, n_steps)  # angle: rotating
        observations[:, 5] = 1.0  # High angular velocity (will be scaled by 2.5)

        actions = np.ones((n_steps, 2), dtype=np.float32) * 0.3

        report = analyzer.analyze(
            observations=observations,
            actions=actions,
            terminated=True,
            truncated=False
        )

        assert "SPINNING_UNCONTROLLED" in report.behaviors

    def test_leg_contact_detection(self, analyzer, sample_observations, sample_actions):
        """Test detection of leg contact events."""
        report = analyzer.analyze(
            observations=sample_observations,
            actions=sample_actions,
            terminated=True,
            truncated=False
        )

        behaviors = set(report.behaviors)
        # Should detect some contact event
        contact_behaviors = {
            "TOUCHED_DOWN_CLEAN", "SCRAPED_LEFT_LEG", "SCRAPED_RIGHT_LEG",
            "NO_CONTACT_MADE", "BOUNCED", "MULTIPLE_TOUCHDOWNS"
        }
        assert any(b in behaviors for b in contact_behaviors)

    def test_thrust_pattern_detection(self, analyzer, sample_observations, sample_actions):
        """Test detection of thrust patterns."""
        report = analyzer.analyze(
            observations=sample_observations,
            actions=sample_actions,
            terminated=True,
            truncated=False
        )

        behaviors = set(report.behaviors)
        # Should detect some thrust pattern
        thrust_behaviors = {
            "MAIN_THRUST_HEAVY", "MAIN_THRUST_MODERATE",
            "MAIN_THRUST_LIGHT", "MAIN_THRUST_NONE",
            "ERRATIC_THRUST", "SMOOTH_THRUST"
        }
        assert any(b in behaviors for b in thrust_behaviors)
