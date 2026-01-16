"""Simulation directory management for data persistence.

Provides immutable, isolated directory structure for simulation data:
- Config snapshot (write-once)
- Per-run JSONL logs (append-only)
- Periodic aggregates
- Model weights
- Charts and text artifacts
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from data.abstractions import SimulationIdentifier

logger = logging.getLogger(__name__)


class SimulationDirectoryError(Exception):
    """Base exception for simulation directory errors."""

    pass


class ConfigAlreadyWrittenError(SimulationDirectoryError):
    """Raised when attempting to write config to a directory that already has one."""

    pass


class DirectoryNotInitializedError(SimulationDirectoryError):
    """Raised when attempting to use a directory that hasn't been initialized."""

    pass


class IsolationViolationError(SimulationDirectoryError):
    """Raised when directory isolation is violated (e.g., directory already exists)."""

    pass


@dataclass
class SimulationPaths:
    """Collection of paths within a simulation directory.

    Attributes:
        root: Root directory for this simulation
        config: Path to config.json
        runs: Path to runs.jsonl
        aggregates: Path to aggregates/ directory
        models: Path to models/ directory
        charts: Path to charts/ directory
        text: Path to text/ directory
    """

    root: Path
    config: Path
    runs: Path
    aggregates: Path
    models: Path
    charts: Path
    text: Path


class SimulationDirectory:
    """Manages an immutable simulation directory.

    Directory structure:
        simulations/{timestamp}_{uuid}/
            config.json          # Immutable config snapshot
            runs.jsonl           # Append-only run log
            aggregates/
                batch_0001.json  # Periodic snapshots
                final.json       # Final aggregates
            models/
                final_model.pt   # Final weights
            charts/
            text/

    Usage:
        sim_id = SimulationIdentifier.create()
        sim_dir = SimulationDirectory(base_path, sim_id)
        sim_dir.initialize()  # Creates directory structure
        sim_dir.write_config(config_dict)  # Write-once
        sim_dir.append_run(run_dict)  # Append-only
    """

    def __init__(
        self,
        base_path: str,
        simulation_id: Optional[SimulationIdentifier] = None,
    ) -> None:
        """Initialize a simulation directory manager.

        Args:
            base_path: Base path for simulations (e.g., 'lunar-lander/')
            simulation_id: Optional identifier; if None, creates a new one
        """
        self._base_path = Path(base_path)
        self._simulation_id = simulation_id or SimulationIdentifier.create()
        self._initialized = False

        # Compute all paths
        root = self._base_path / "simulations" / self._simulation_id.directory_name
        self._paths = SimulationPaths(
            root=root,
            config=root / "config.json",
            runs=root / "runs.jsonl",
            aggregates=root / "aggregates",
            models=root / "models",
            charts=root / "charts",
            text=root / "text",
        )

    @property
    def simulation_id(self) -> SimulationIdentifier:
        """Return the simulation identifier."""
        return self._simulation_id

    @property
    def paths(self) -> SimulationPaths:
        """Return the paths object for this simulation."""
        return self._paths

    @property
    def root_path(self) -> Path:
        """Return the root directory path."""
        return self._paths.root

    @property
    def config_path(self) -> Path:
        """Return the config.json path."""
        return self._paths.config

    @property
    def runs_path(self) -> Path:
        """Return the runs.jsonl path."""
        return self._paths.runs

    @property
    def aggregates_path(self) -> Path:
        """Return the aggregates/ directory path."""
        return self._paths.aggregates

    @property
    def models_path(self) -> Path:
        """Return the models/ directory path."""
        return self._paths.models

    @property
    def charts_path(self) -> Path:
        """Return the charts/ directory path."""
        return self._paths.charts

    @property
    def text_path(self) -> Path:
        """Return the text/ directory path."""
        return self._paths.text

    def verify_isolation(self) -> None:
        """Verify that this simulation directory doesn't already exist.

        Raises:
            IsolationViolationError: If directory already exists
        """
        if self._paths.root.exists():
            raise IsolationViolationError(
                f"Simulation directory already exists: {self._paths.root}. "
                "Each simulation must use a fresh directory."
            )

    def initialize(self, verify_isolation: bool = True) -> None:
        """Create the directory structure for this simulation.

        Args:
            verify_isolation: If True, verify directory doesn't exist first

        Raises:
            IsolationViolationError: If verify_isolation is True and directory exists
        """
        if verify_isolation:
            self.verify_isolation()

        # Create all directories
        self._paths.root.mkdir(parents=True, exist_ok=True)
        self._paths.aggregates.mkdir(exist_ok=True)
        self._paths.models.mkdir(exist_ok=True)
        self._paths.charts.mkdir(exist_ok=True)
        self._paths.text.mkdir(exist_ok=True)

        self._initialized = True
        logger.info(f"Initialized simulation directory: {self._paths.root}")

    def _ensure_initialized(self) -> None:
        """Ensure the directory has been initialized.

        Raises:
            DirectoryNotInitializedError: If initialize() hasn't been called
        """
        if not self._initialized and not self._paths.root.exists():
            raise DirectoryNotInitializedError(
                f"Simulation directory not initialized: {self._paths.root}. "
                "Call initialize() first."
            )

    def write_config(self, config_dict: Dict[str, Any]) -> None:
        """Write config snapshot to config.json (write-once semantics).

        Args:
            config_dict: Configuration dictionary to serialize

        Raises:
            ConfigAlreadyWrittenError: If config.json already exists
            DirectoryNotInitializedError: If directory not initialized
        """
        self._ensure_initialized()

        if self._paths.config.exists():
            raise ConfigAlreadyWrittenError(
                f"Config already written: {self._paths.config}. "
                "Config is immutable and can only be written once."
            )

        with open(self._paths.config, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Config snapshot written to {self._paths.config}")

    def read_config(self) -> Dict[str, Any]:
        """Read the config snapshot.

        Returns:
            The configuration dictionary

        Raises:
            FileNotFoundError: If config hasn't been written
        """
        with open(self._paths.config, "r", encoding="utf-8") as f:
            return json.load(f)

    def append_run(self, run_dict: Dict[str, Any]) -> None:
        """Append a run record to runs.jsonl (append-only).

        Args:
            run_dict: Run record dictionary to serialize

        Raises:
            DirectoryNotInitializedError: If directory not initialized
        """
        self._ensure_initialized()

        with open(self._paths.runs, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_dict, ensure_ascii=False) + "\n")

    def read_runs(self) -> list:
        """Read all run records from runs.jsonl.

        Returns:
            List of run record dictionaries
        """
        if not self._paths.runs.exists():
            return []

        runs = []
        with open(self._paths.runs, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
        return runs

    def write_aggregate(self, filename: str, aggregate_dict: Dict[str, Any]) -> None:
        """Write an aggregate snapshot to the aggregates directory.

        Args:
            filename: Filename for the aggregate (e.g., 'batch_0001.json')
            aggregate_dict: Aggregate data dictionary
        """
        self._ensure_initialized()

        path = self._paths.aggregates / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(aggregate_dict, f, indent=2, ensure_ascii=False)

        logger.debug(f"Aggregate written to {path}")

    def read_aggregate(self, filename: str) -> Optional[Dict[str, Any]]:
        """Read an aggregate snapshot.

        Args:
            filename: Filename of the aggregate to read

        Returns:
            The aggregate dictionary, or None if not found
        """
        path = self._paths.aggregates / filename
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def from_existing(cls, simulation_path: str) -> "SimulationDirectory":
        """Load an existing simulation directory.

        Args:
            simulation_path: Full path to an existing simulation directory

        Returns:
            SimulationDirectory instance for the existing directory
        """
        path = Path(simulation_path)
        if not path.exists():
            raise DirectoryNotInitializedError(
                f"Simulation directory does not exist: {path}"
            )

        # Parse the simulation ID from directory name
        sim_id = SimulationIdentifier.from_string(path.name)

        # Determine base path (parent of 'simulations' directory)
        base_path = path.parent.parent

        instance = cls(str(base_path), sim_id)
        instance._initialized = True
        return instance
