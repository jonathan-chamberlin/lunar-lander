"""Output formatting utilities for behavior reports.

This module provides formatting functions for displaying behavior analysis
results in a human-readable format.
"""

from typing import TYPE_CHECKING

from constants import BEHAVIOR_CATEGORIES, BEHAVIOR_SHORT_NAMES

if TYPE_CHECKING:
    from analysis.behavior_analysis import BehaviorReport


def format_behavior_output(behavior_report: "BehaviorReport") -> str:
    """Format behavior report into categorized lines with icons.

    Groups behaviors by category (horizontal, vertical, orientation, thrust, contact)
    and displays them with appropriate icons and shortened names.

    Args:
        behavior_report: BehaviorReport containing detected behaviors

    Returns:
        Formatted multi-line string with categorized behaviors
    """
    lines = []

    # Group behaviors by category
    categorized = {cat: [] for cat in BEHAVIOR_CATEGORIES}
    uncategorized = []

    for behavior in behavior_report.behaviors:
        found = False
        for cat_name, cat_info in BEHAVIOR_CATEGORIES.items():
            if behavior in cat_info['behaviors']:
                short_name = BEHAVIOR_SHORT_NAMES.get(behavior, behavior)
                categorized[cat_name].append(short_name)
                found = True
                break
        if not found:
            short_name = BEHAVIOR_SHORT_NAMES.get(behavior, behavior)
            uncategorized.append(short_name)

    # Build output lines for non-empty categories
    for cat_name in ['horizontal', 'vertical', 'orientation', 'thrust', 'contact']:
        if categorized[cat_name]:
            icon = BEHAVIOR_CATEGORIES[cat_name]['icon']
            behaviors_str = ', '.join(categorized[cat_name])
            lines.append(f"    {icon} {behaviors_str}")

    if uncategorized:
        lines.append(f"    📋 {', '.join(uncategorized)}")

    return '\n'.join(lines)
