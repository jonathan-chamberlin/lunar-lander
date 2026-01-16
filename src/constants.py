"""Shared constants for the Lunar Lander project.

This module consolidates behavior definitions, outcome mappings, and display
constants used across main.py, analysis/diagnostics.py, and analysis/charts.py.
"""

from typing import Dict, Set, List

# =============================================================================
# Behavior Category Mappings (for pretty printing)
# =============================================================================

BEHAVIOR_CATEGORIES: Dict[str, Dict] = {
    'horizontal': {
        'icon': '↔',
        'behaviors': {'STAYED_CENTERED', 'DRIFTED_LEFT', 'DRIFTED_RIGHT', 'RETURNED_TO_CENTER',
                      'HORIZONTAL_OSCILLATION', 'STRONG_LATERAL_VELOCITY'}
    },
    'vertical': {
        'icon': '↕',
        'behaviors': {'CONTROLLED_DESCENT', 'SLOW_DESCENT', 'RAPID_DESCENT', 'STABLE_ALTITUDE',
                      'ASCENDED', 'YO_YO_PATTERN', 'CONTINUOUS_BURN', 'LATE_BRAKING', 'FREEFALL',
                      'STALLED_THEN_FELL', 'DIRECT_APPROACH', 'CURVED_APPROACH', 'SPIRAL_DESCENT',
                      'ZIGZAG_DESCENT', 'SUICIDE_BURN', 'GRADUAL_SLOWDOWN', 'HOVER_NEAR_GROUND_TIMEOUT'}
    },
    'orientation': {
        'icon': '↻',
        'behaviors': {'STAYED_UPRIGHT', 'SLIGHT_LEFT_LEAN', 'SLIGHT_RIGHT_LEAN', 'HEAVY_LEFT_TILT',
                      'HEAVY_RIGHT_TILT', 'FLIPPED_OVER', 'EXCEEDED_SPIN_RATE', 'RECOVERED_FROM_TILT',
                      'PROGRESSIVE_TILT', 'WOBBLING', 'NEVER_STABILIZED', 'CONTROLLED_THROUGHOUT',
                      'LOST_CONTROL_LATE', 'OVERCORRECTED_TO_CRASH'}
    },
    'thrust': {
        'icon': '🔥',
        'behaviors': {'MAIN_THRUST_HEAVY', 'MAIN_THRUST_MODERATE', 'MAIN_THRUST_LIGHT', 'MAIN_THRUST_NONE',
                      'SIDE_THRUST_LEFT_BIAS', 'SIDE_THRUST_RIGHT_BIAS', 'SIDE_THRUST_BALANCED',
                      'ERRATIC_THRUST', 'SMOOTH_THRUST', 'FULL_THROTTLE_BURST'}
    },
    'contact': {
        'icon': '👣',
        'behaviors': {'NO_CONTACT_MADE', 'SCRAPED_LEFT_LEG', 'SCRAPED_RIGHT_LEG', 'TOUCHED_DOWN_CLEAN',
                      'BOUNCED', 'MULTIPLE_TOUCHDOWNS', 'PROLONGED_ONE_LEG', 'REACHED_LOW_ALTITUDE',
                      'STAYED_HIGH', 'PEAKED_ABOVE_START', 'GROUND_APPROACH_ABORT'}
    },
}

# =============================================================================
# Landing Outcomes
# =============================================================================

# Outcomes that indicate a safe landing (touched down without crash)
# Note: LANDED_HARD, LANDED_TILTED, TOUCHED_DOWN_ONE_LEG are excluded because
# OUTCOME_TO_CATEGORY maps them to 'crashed' - this keeps the indicator consistent
SAFE_LANDING_OUTCOMES: Set[str] = {
    'LANDED_PERFECTLY', 'LANDED_SOFTLY', 'LANDED_WITH_DRIFT', 'TIMED_OUT_ON_GROUND'
}

# =============================================================================
# Outcome Categories
# =============================================================================

# Mapping from category name to specific outcomes (used in diagnostics)
OUTCOME_CATEGORIES: Dict[str, List[str]] = {
    'landed': ['LANDED_PERFECTLY', 'LANDED_SOFTLY', 'LANDED_HARD', 'LANDED_TILTED', 'TOUCHED_DOWN_ONE_LEG'],
    'crashed': ['CRASHED_HIGH_VELOCITY', 'CRASHED_TILTED', 'CRASHED_SIDEWAYS', 'CRASHED_SPINNING'],
    'timed_out': ['TIMED_OUT_HOVERING', 'TIMED_OUT_DESCENDING', 'TIMED_OUT_ASCENDING'],
    'flew_off': ['FLEW_OFF_TOP', 'FLEW_OFF_LEFT', 'FLEW_OFF_RIGHT', 'FLEW_OFF_LEFT_TILTED', 'FLEW_OFF_RIGHT_TILTED'],
    'bouncing': ['SUSTAINED_BOUNCING'],
}

# List of category names in display order (used in charts)
OUTCOME_CATEGORY_ORDER: List[str] = ['landed', 'crashed', 'timed_out', 'flew_off', 'bouncing']

# Mapping from specific outcome to category (used in charts)
# "Landed" = proper landings only (upright, controlled)
# "Crashed" = impacts including bad touchdowns (one leg, tilted, hard)
OUTCOME_TO_CATEGORY: Dict[str, str] = {
    # Good landings
    'LANDED_PERFECTLY': 'landed',
    'LANDED_SOFTLY': 'landed',
    'LANDED_WITH_DRIFT': 'landed',
    'TIMED_OUT_ON_GROUND': 'landed',
    # Bad touchdowns count as crashes
    'LANDED_HARD': 'crashed',
    'LANDED_TILTED': 'crashed',
    'TOUCHED_DOWN_ONE_LEG': 'crashed',
    # Actual crashes
    'CRASHED_HIGH_VELOCITY': 'crashed',
    'CRASHED_TILTED': 'crashed',
    'CRASHED_SIDEWAYS': 'crashed',
    'CRASHED_SPINNING': 'crashed',
    'CRASHED_OTHER': 'crashed',
    # Timeouts
    'TIMED_OUT_HOVERING': 'timed_out',
    'TIMED_OUT_DESCENDING': 'timed_out',
    'TIMED_OUT_ASCENDING': 'timed_out',
    # Flew off screen
    'FLEW_OFF_TOP': 'flew_off',
    'FLEW_OFF_LEFT': 'flew_off',
    'FLEW_OFF_RIGHT': 'flew_off',
    'FLEW_OFF_LEFT_TILTED': 'flew_off',
    'FLEW_OFF_RIGHT_TILTED': 'flew_off',
    # Sustained bouncing (exploitative behavior)
    'SUSTAINED_BOUNCING': 'bouncing',
}

# Colors for outcome categories in charts
OUTCOME_COLORS: Dict[str, str] = {
    'landed': '#2ecc71',     # Green
    'crashed': '#e74c3c',    # Red
    'timed_out': '#f39c12',  # Yellow/Orange
    'flew_off': '#3498db',   # Blue
    'bouncing': '#9b59b6',   # Purple
}

# =============================================================================
# Quality Behaviors (for analysis)
# =============================================================================

QUALITY_BEHAVIORS: Dict[str, List[str]] = {
    'good': ['STAYED_UPRIGHT', 'STAYED_CENTERED', 'CONTROLLED_DESCENT', 'CONTROLLED_THROUGHOUT',
             'SMOOTH_THRUST', 'TOUCHED_DOWN_CLEAN', 'DIRECT_APPROACH', 'GRADUAL_SLOWDOWN',
             'REACHED_LOW_ALTITUDE', 'RECOVERED_FROM_TILT', 'RETURNED_TO_CENTER'],
    'bad': ['NEVER_STABILIZED', 'LOST_CONTROL_LATE', 'OVERCORRECTED_TO_CRASH', 'EXCEEDED_SPIN_RATE',
            'FLIPPED_OVER', 'ERRATIC_THRUST', 'FREEFALL', 'NO_CONTACT_MADE', 'STAYED_HIGH'],
}

# Behaviors displayed in report card chart
REPORT_CARD_BEHAVIORS: List[str] = [
    # Good behaviors (should increase)
    'STAYED_UPRIGHT',
    'STAYED_CENTERED',
    'CONTROLLED_DESCENT',
    'CONTROLLED_THROUGHOUT',
    'TOUCHED_DOWN_CLEAN',
    'REACHED_LOW_ALTITUDE',
    # Bad behaviors (should decrease)
    'NEVER_STABILIZED',
    'EXCEEDED_SPIN_RATE',
    'FREEFALL',
    'ERRATIC_THRUST',
    'CRASHED_HIGH_VELOCITY',
    'FLIPPED_OVER',
]

# =============================================================================
# Short Names for Behaviors (compact display)
# =============================================================================

BEHAVIOR_SHORT_NAMES: Dict[str, str] = {
    'STAYED_CENTERED': 'CENTERED', 'DRIFTED_LEFT': 'DRIFT_L', 'DRIFTED_RIGHT': 'DRIFT_R',
    'RETURNED_TO_CENTER': 'RETURNED', 'HORIZONTAL_OSCILLATION': 'H_OSCILLATE',
    'STRONG_LATERAL_VELOCITY': 'STRONG_LATERAL',
    'CONTROLLED_DESCENT': 'CTRL_DESC', 'SLOW_DESCENT': 'SLOW_DESC', 'RAPID_DESCENT': 'RAPID_DESC',
    'STABLE_ALTITUDE': 'HOVER', 'YO_YO_PATTERN': 'YO_YO', 'CONTINUOUS_BURN': 'CONT_BURN',
    'LATE_BRAKING': 'LATE_BRAKE', 'STALLED_THEN_FELL': 'STALLED',
    'STAYED_UPRIGHT': 'UPRIGHT', 'SLIGHT_LEFT_LEAN': 'LEAN_L', 'SLIGHT_RIGHT_LEAN': 'LEAN_R',
    'HEAVY_LEFT_TILT': 'TILT_L', 'HEAVY_RIGHT_TILT': 'TILT_R', 'FLIPPED_OVER': 'FLIPPED',
    'EXCEEDED_SPIN_RATE': 'SPINNING', 'RECOVERED_FROM_TILT': 'RECOVERED',
    'PROGRESSIVE_TILT': 'PROG_TILT', 'NEVER_STABILIZED': 'UNSTABLE',
    'CONTROLLED_THROUGHOUT': 'CONTROLLED', 'LOST_CONTROL_LATE': 'LOST_CTRL',
    'OVERCORRECTED_TO_CRASH': 'OVERCORRECT',
    'MAIN_THRUST_HEAVY': 'MAIN_HEAVY', 'MAIN_THRUST_MODERATE': 'MAIN_MOD',
    'MAIN_THRUST_LIGHT': 'MAIN_LIGHT', 'MAIN_THRUST_NONE': 'MAIN_NONE',
    'SIDE_THRUST_LEFT_BIAS': 'SIDE_L', 'SIDE_THRUST_RIGHT_BIAS': 'SIDE_R',
    'SIDE_THRUST_BALANCED': 'SIDE_BAL', 'ERRATIC_THRUST': 'ERRATIC', 'SMOOTH_THRUST': 'SMOOTH',
    'FULL_THROTTLE_BURST': 'FULL_THROTTLE',
    'NO_CONTACT_MADE': 'NO_CONTACT', 'SCRAPED_LEFT_LEG': 'SCRAPE_L', 'SCRAPED_RIGHT_LEG': 'SCRAPE_R',
    'TOUCHED_DOWN_CLEAN': 'CLEAN_TD', 'MULTIPLE_TOUCHDOWNS': 'MULTI_TD', 'PROLONGED_ONE_LEG': 'ONE_LEG',
    'REACHED_LOW_ALTITUDE': 'LOW_ALT', 'STAYED_HIGH': 'HIGH', 'PEAKED_ABOVE_START': 'PEAKED',
    'GROUND_APPROACH_ABORT': 'ABORTED', 'DIRECT_APPROACH': 'DIRECT', 'CURVED_APPROACH': 'CURVED',
    'SPIRAL_DESCENT': 'SPIRAL', 'ZIGZAG_DESCENT': 'ZIGZAG', 'SUICIDE_BURN': 'SUICIDE',
    'GRADUAL_SLOWDOWN': 'GRADUAL', 'HOVERED_OVER_GOAL_TIMEOUT': 'HOVER_EXPLOIT',
    'HOVER_NEAR_GROUND_TIMEOUT': 'HOVER_GROUND',
}
