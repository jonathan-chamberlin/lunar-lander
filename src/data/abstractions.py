"""Formal abstractions for data architecture.

Defines the core identifiers for the three-level abstraction hierarchy:
- Simulation: One execution of main.py (owns all learning state)
- Run: One episode within a simulation
- Experiment: Collection of simulations for comparison
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid


@dataclass(frozen=True)
class SimulationIdentifier:
    """Unique identifier for a simulation (one execution of main.py).

    A simulation owns all learning state and consists of multiple runs (episodes).
    The identifier combines a timestamp and short UUID for uniqueness and human readability.

    Attributes:
        timestamp: ISO 8601 formatted timestamp of simulation start
        uuid_short: First 8 characters of a UUID4 for uniqueness
    """

    timestamp: str
    uuid_short: str

    @classmethod
    def create(cls) -> "SimulationIdentifier":
        """Create a new simulation identifier with current timestamp and UUID."""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        uuid_short = uuid.uuid4().hex[:8]
        return cls(timestamp=timestamp, uuid_short=uuid_short)

    @classmethod
    def from_string(cls, identifier: str) -> "SimulationIdentifier":
        """Parse a simulation identifier from its string representation.

        Args:
            identifier: String in format 'YYYY-MM-DDTHH-MM-SS_xxxxxxxx'

        Returns:
            Parsed SimulationIdentifier

        Raises:
            ValueError: If the string format is invalid
        """
        parts = identifier.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid simulation identifier format: {identifier}")
        return cls(timestamp=parts[0], uuid_short=parts[1])

    def __str__(self) -> str:
        """Return the canonical string representation."""
        return f"{self.timestamp}_{self.uuid_short}"

    @property
    def directory_name(self) -> str:
        """Return the directory name for this simulation."""
        return str(self)


@dataclass(frozen=True)
class RunIdentifier:
    """Unique identifier for a run (one episode within a simulation).

    A run is a single episode that produces rewards, steps, and outcomes.
    It is uniquely identified by its simulation and run number.

    Attributes:
        simulation_id: Parent simulation identifier
        run_number: Zero-indexed episode number within the simulation
    """

    simulation_id: SimulationIdentifier
    run_number: int

    def __post_init__(self) -> None:
        """Validate run number is non-negative."""
        if self.run_number < 0:
            raise ValueError(f"Run number must be non-negative, got {self.run_number}")

    def __str__(self) -> str:
        """Return the canonical string representation."""
        return f"{self.simulation_id}/run_{self.run_number:05d}"


@dataclass
class ExperimentDefinition:
    """Definition of an experiment (collection of simulations for comparison).

    An experiment groups related simulations together for comparative analysis.
    It stores only metadata and references, not raw simulation data.

    Attributes:
        name: Human-readable experiment name (used as directory name)
        description: Optional description of the experiment purpose
        simulation_ids: List of simulation identifiers included in this experiment
        created_at: ISO 8601 timestamp when the experiment was created
        tags: Optional list of tags for filtering and organization
    """

    name: str
    description: Optional[str] = None
    simulation_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate experiment name is valid as a directory name."""
        invalid_chars = set('<>:"/\\|?*')
        if any(char in self.name for char in invalid_chars):
            raise ValueError(
                f"Experiment name contains invalid characters: {self.name}"
            )
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty or whitespace")

    def add_simulation(self, simulation_id: SimulationIdentifier) -> None:
        """Add a simulation to this experiment.

        Args:
            simulation_id: The simulation identifier to add
        """
        id_str = str(simulation_id)
        if id_str not in self.simulation_ids:
            self.simulation_ids.append(id_str)

    def remove_simulation(self, simulation_id: SimulationIdentifier) -> bool:
        """Remove a simulation from this experiment.

        Args:
            simulation_id: The simulation identifier to remove

        Returns:
            True if the simulation was removed, False if not found
        """
        id_str = str(simulation_id)
        if id_str in self.simulation_ids:
            self.simulation_ids.remove(id_str)
            return True
        return False

    @property
    def simulation_count(self) -> int:
        """Return the number of simulations in this experiment."""
        return len(self.simulation_ids)
