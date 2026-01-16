"""Data architecture package for simulation, run, and experiment management.

This package provides:
- Formal abstractions for Simulation, Run, and Experiment identifiers
- Directory management for simulation isolation
- Config snapshot serialization
- Per-run JSONL logging
- Periodic aggregate persistence
- Experiment aggregation
"""

from data.abstractions import (
    SimulationIdentifier,
    RunIdentifier,
    ExperimentDefinition,
)
from data.simulation_io import SimulationDirectory
from data.config_serializer import serialize_config, create_config_snapshot
from data.run_logger import RunRecord, RunLogger
from data.aggregate_writer import AggregateWriter
from data.experiment_io import ExperimentDirectory

__all__ = [
    # Abstractions
    "SimulationIdentifier",
    "RunIdentifier",
    "ExperimentDefinition",
    # Simulation I/O
    "SimulationDirectory",
    # Config
    "serialize_config",
    "create_config_snapshot",
    # Run logging
    "RunRecord",
    "RunLogger",
    # Aggregates
    "AggregateWriter",
    # Experiments
    "ExperimentDirectory",
]
