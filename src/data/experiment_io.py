"""Experiment directory management for cross-simulation analysis.

An experiment is a collection of simulations grouped for comparison.
Experiments store only metadata and aggregated summaries, not raw data.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.abstractions import ExperimentDefinition, SimulationIdentifier

logger = logging.getLogger(__name__)


class ExperimentDirectoryError(Exception):
    """Base exception for experiment directory errors."""

    pass


class ExperimentExistsError(ExperimentDirectoryError):
    """Raised when attempting to create an experiment that already exists."""

    pass


class ExperimentNotFoundError(ExperimentDirectoryError):
    """Raised when an experiment is not found."""

    pass


class ExperimentDirectory:
    """Manages an experiment directory.

    Directory structure:
        experiments/{name}/
            experiment.json    # Definition + simulation references
            summary.json       # Aggregated summary across simulations

    Experiments are metadata-only - they reference simulations but don't
    duplicate raw data. Aggregation is performed by loading final.json
    from each referenced simulation.

    Usage:
        exp_dir = ExperimentDirectory(base_path, "lr_sweep")
        exp_dir.initialize(description="Learning rate comparison")
        exp_dir.add_simulation(sim_id)
        exp_dir.compute_summary(simulations_path)
    """

    def __init__(
        self,
        base_path: str,
        experiment_name: str,
    ) -> None:
        """Initialize an experiment directory manager.

        Args:
            base_path: Base path for experiments (e.g., 'lunar-lander/')
            experiment_name: Name of the experiment (used as directory name)
        """
        self._base_path = Path(base_path)
        self._experiment_name = experiment_name
        self._definition: Optional[ExperimentDefinition] = None

        # Compute paths
        self._root = self._base_path / "experiments" / experiment_name
        self._experiment_path = self._root / "experiment.json"
        self._summary_path = self._root / "summary.json"

    @property
    def experiment_name(self) -> str:
        """Return the experiment name."""
        return self._experiment_name

    @property
    def root_path(self) -> Path:
        """Return the root directory path."""
        return self._root

    @property
    def experiment_path(self) -> Path:
        """Return the experiment.json path."""
        return self._experiment_path

    @property
    def summary_path(self) -> Path:
        """Return the summary.json path."""
        return self._summary_path

    @property
    def definition(self) -> Optional[ExperimentDefinition]:
        """Return the experiment definition."""
        return self._definition

    def exists(self) -> bool:
        """Check if this experiment already exists."""
        return self._experiment_path.exists()

    def initialize(
        self,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        allow_existing: bool = False,
    ) -> None:
        """Create the experiment directory and definition.

        Args:
            description: Optional experiment description
            tags: Optional list of tags for organization
            allow_existing: If True, load existing experiment instead of error

        Raises:
            ExperimentExistsError: If experiment exists and allow_existing is False
        """
        if self.exists():
            if allow_existing:
                self.load()
                return
            raise ExperimentExistsError(
                f"Experiment already exists: {self._experiment_name}"
            )

        self._root.mkdir(parents=True, exist_ok=True)

        self._definition = ExperimentDefinition(
            name=self._experiment_name,
            description=description,
            simulation_ids=[],
            tags=tags or [],
        )

        self._save_definition()
        logger.info(f"Initialized experiment: {self._experiment_name}")

    def load(self) -> None:
        """Load an existing experiment definition.

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist
        """
        if not self.exists():
            raise ExperimentNotFoundError(
                f"Experiment not found: {self._experiment_name}"
            )

        with open(self._experiment_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._definition = ExperimentDefinition(
            name=data["name"],
            description=data.get("description"),
            simulation_ids=data.get("simulation_ids", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            tags=data.get("tags", []),
        )

    def _save_definition(self) -> None:
        """Save the current definition to experiment.json."""
        if self._definition is None:
            return

        data = {
            "name": self._definition.name,
            "description": self._definition.description,
            "simulation_ids": self._definition.simulation_ids,
            "created_at": self._definition.created_at,
            "tags": self._definition.tags,
        }

        with open(self._experiment_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_simulation(self, simulation_id: SimulationIdentifier) -> None:
        """Add a simulation to this experiment.

        Args:
            simulation_id: The simulation identifier to add
        """
        if self._definition is None:
            raise ExperimentNotFoundError("Experiment not initialized")

        self._definition.add_simulation(simulation_id)
        self._save_definition()
        logger.info(f"Added simulation {simulation_id} to experiment {self._experiment_name}")

    def remove_simulation(self, simulation_id: SimulationIdentifier) -> bool:
        """Remove a simulation from this experiment.

        Args:
            simulation_id: The simulation identifier to remove

        Returns:
            True if removed, False if not found
        """
        if self._definition is None:
            raise ExperimentNotFoundError("Experiment not initialized")

        removed = self._definition.remove_simulation(simulation_id)
        if removed:
            self._save_definition()
            logger.info(f"Removed simulation {simulation_id} from experiment {self._experiment_name}")
        return removed

    def get_simulation_ids(self) -> List[str]:
        """Get list of simulation IDs in this experiment.

        Returns:
            List of simulation ID strings
        """
        if self._definition is None:
            return []
        return self._definition.simulation_ids.copy()

    def compute_summary(
        self,
        simulations_base_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Compute and write aggregated summary from simulation data.

        Loads final.json from each referenced simulation and computes
        cross-simulation statistics.

        Args:
            simulations_base_path: Base path for simulations (default: base_path)

        Returns:
            The computed summary dictionary
        """
        if self._definition is None:
            raise ExperimentNotFoundError("Experiment not initialized")

        sim_base = simulations_base_path or (self._base_path / "simulations")

        # Load aggregates from each simulation
        simulation_data = []
        for sim_id_str in self._definition.simulation_ids:
            sim_path = sim_base / sim_id_str
            final_path = sim_path / "aggregates" / "final.json"

            if final_path.exists():
                with open(final_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["_simulation_id"] = sim_id_str
                    simulation_data.append(data)
            else:
                logger.warning(f"No final.json found for simulation: {sim_id_str}")

        summary = self._aggregate_simulations(simulation_data)

        # Write summary
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Experiment summary written to {self._summary_path}")
        return summary

    def _aggregate_simulations(
        self,
        simulation_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate data from multiple simulations.

        Args:
            simulation_data: List of final.json dictionaries from simulations

        Returns:
            Aggregated summary dictionary
        """
        if not simulation_data:
            return {
                "_metadata": {
                    "schema_version": "1.0.0",
                    "experiment_name": self._experiment_name,
                    "simulation_count": 0,
                    "created_at": datetime.now().isoformat(),
                },
                "error": "No simulation data available",
            }

        # Extract metrics from each simulation
        success_rates = []
        mean_rewards = []
        max_rewards = []
        total_episodes = []
        max_streaks = []

        for data in simulation_data:
            summary = data.get("summary", {})
            streaks = data.get("streaks", {})

            if "success_rate" in summary:
                success_rates.append(summary["success_rate"])
            if "mean_reward" in summary:
                mean_rewards.append(summary["mean_reward"])
            if "max_reward" in summary:
                max_rewards.append(summary["max_reward"])
            if "total_episodes" in summary:
                total_episodes.append(summary["total_episodes"])
            if "max_streak" in streaks:
                max_streaks.append(streaks["max_streak"])

        # Compute cross-simulation statistics
        def safe_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            import statistics
            return {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }

        summary = {
            "_metadata": {
                "schema_version": "1.0.0",
                "experiment_name": self._experiment_name,
                "simulation_count": len(simulation_data),
                "simulation_ids": [d["_simulation_id"] for d in simulation_data],
                "created_at": datetime.now().isoformat(),
            },
            "aggregate_metrics": {
                "success_rate": safe_stats(success_rates),
                "mean_reward": safe_stats(mean_rewards),
                "max_reward": safe_stats(max_rewards),
                "total_episodes": safe_stats([float(x) for x in total_episodes]),
                "max_streak": safe_stats([float(x) for x in max_streaks]),
            },
            "per_simulation": [
                {
                    "simulation_id": d["_simulation_id"],
                    "summary": d.get("summary", {}),
                    "streaks": d.get("streaks", {}),
                }
                for d in simulation_data
            ],
        }

        # Add best simulation identification
        if success_rates:
            best_idx = success_rates.index(max(success_rates))
            summary["best_simulation"] = {
                "by_success_rate": simulation_data[best_idx]["_simulation_id"],
                "success_rate": success_rates[best_idx],
            }

        if mean_rewards:
            best_idx = mean_rewards.index(max(mean_rewards))
            summary["best_simulation"]["by_mean_reward"] = simulation_data[best_idx]["_simulation_id"]
            summary["best_simulation"]["mean_reward"] = mean_rewards[best_idx]

        return summary

    def read_summary(self) -> Optional[Dict[str, Any]]:
        """Read the experiment summary.

        Returns:
            Summary dictionary, or None if not available
        """
        if not self._summary_path.exists():
            return None

        with open(self._summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def list_experiments(cls, base_path: str) -> List[str]:
        """List all experiment names in the base path.

        Args:
            base_path: Base path containing experiments/ directory

        Returns:
            List of experiment names
        """
        experiments_path = Path(base_path) / "experiments"
        if not experiments_path.exists():
            return []

        return [
            d.name
            for d in experiments_path.iterdir()
            if d.is_dir() and (d / "experiment.json").exists()
        ]
