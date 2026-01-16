"""Configuration snapshot serialization for simulation reproducibility.

Provides serialization of frozen dataclasses with metadata for config snapshots.
"""

import dataclasses
from datetime import datetime
from typing import Any, Dict, Tuple

from data.abstractions import SimulationIdentifier


def serialize_config(config: Any) -> Dict[str, Any]:
    """Recursively serialize a configuration object to a dictionary.

    Handles:
    - Frozen dataclasses (recursively serialized)
    - Regular dataclasses
    - Tuples (converted to lists for JSON compatibility)
    - Primitive types (passed through)

    Args:
        config: Configuration object (typically a dataclass)

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        # Recursively serialize dataclass fields
        result = {}
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            result[field.name] = serialize_config(value)
        return result

    elif isinstance(config, tuple):
        # Convert tuples to lists for JSON compatibility
        return [serialize_config(item) for item in config]

    elif isinstance(config, list):
        return [serialize_config(item) for item in config]

    elif isinstance(config, dict):
        return {key: serialize_config(value) for key, value in config.items()}

    else:
        # Primitive types: int, float, str, bool, None
        return config


def create_config_snapshot(
    config: Any,
    simulation_id: SimulationIdentifier,
    version: str = "1.0.0",
) -> Dict[str, Any]:
    """Create a complete configuration snapshot with metadata.

    The snapshot includes:
    - All configuration values (recursively serialized)
    - Simulation identifier
    - Timestamp
    - Schema version

    Args:
        config: Configuration object to snapshot
        simulation_id: The simulation identifier
        version: Schema version string for future compatibility

    Returns:
        Complete snapshot dictionary ready for persistence
    """
    return {
        "_metadata": {
            "schema_version": version,
            "simulation_id": str(simulation_id),
            "created_at": datetime.now().isoformat(),
            "config_type": type(config).__name__,
        },
        "config": serialize_config(config),
    }


def extract_config_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the configuration dictionary from a snapshot.

    Args:
        snapshot: Complete snapshot dictionary

    Returns:
        Just the configuration portion
    """
    return snapshot.get("config", {})


def get_snapshot_metadata(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from a snapshot.

    Args:
        snapshot: Complete snapshot dictionary

    Returns:
        The metadata dictionary
    """
    return snapshot.get("_metadata", {})
