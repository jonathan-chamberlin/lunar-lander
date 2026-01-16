"""Per-run JSONL logging for simulation data.

Provides append-only logging of individual run (episode) records to runs.jsonl.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Complete record of a single run (episode).

    Contains all per-episode metrics for raw data storage.
    This is the RAW tier of data - immutable and complete.

    Attributes:
        run_number: Zero-indexed episode number
        env_reward: Raw environment reward (no shaping)
        shaped_bonus: Additional reward from shaping
        total_reward: env_reward + shaped_bonus
        steps: Number of steps in the episode
        duration_seconds: Wall-clock time for the episode
        success: Whether reward >= success_threshold
        outcome: Outcome classification (e.g., 'LANDED_SAFE', 'CRASHED')
        behaviors: List of detected behaviors
        terminated: Whether episode ended due to terminal state
        truncated: Whether episode was truncated (time limit)
        rendered: Whether this episode was rendered
        timestamp: ISO 8601 timestamp when run completed
    """

    run_number: int
    env_reward: float
    shaped_bonus: float
    total_reward: float
    steps: int
    duration_seconds: float
    success: bool
    outcome: str
    behaviors: List[str]
    terminated: bool
    truncated: bool
    rendered: bool
    timestamp: str

    @classmethod
    def create(
        cls,
        run_number: int,
        env_reward: float,
        shaped_bonus: float,
        steps: int,
        duration_seconds: float,
        success: bool,
        outcome: str,
        behaviors: List[str],
        terminated: bool,
        truncated: bool,
        rendered: bool,
    ) -> "RunRecord":
        """Create a RunRecord with automatic timestamp.

        Args:
            run_number: Zero-indexed episode number
            env_reward: Raw environment reward
            shaped_bonus: Additional reward from shaping
            steps: Number of steps
            duration_seconds: Wall-clock time
            success: Whether successful
            outcome: Outcome classification
            behaviors: List of detected behaviors
            terminated: Whether terminated
            truncated: Whether truncated
            rendered: Whether rendered

        Returns:
            New RunRecord instance
        """
        return cls(
            run_number=run_number,
            env_reward=env_reward,
            shaped_bonus=shaped_bonus,
            total_reward=env_reward + shaped_bonus,
            steps=steps,
            duration_seconds=duration_seconds,
            success=success,
            outcome=outcome,
            behaviors=behaviors,
            terminated=terminated,
            truncated=truncated,
            rendered=rendered,
            timestamp=datetime.now().isoformat(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RunLogger:
    """Append-only logger for per-run records.

    Writes RunRecord instances to a JSONL file (one JSON object per line).
    This provides efficient append-only logging without loading the entire file.

    Usage:
        logger = RunLogger(sim_dir.runs_path)
        logger.log_run(run_record)
    """

    def __init__(self, runs_path: Path) -> None:
        """Initialize the run logger.

        Args:
            runs_path: Path to the runs.jsonl file
        """
        self._runs_path = Path(runs_path)
        self._run_count = 0

    @property
    def runs_path(self) -> Path:
        """Return the path to the runs.jsonl file."""
        return self._runs_path

    @property
    def run_count(self) -> int:
        """Return the number of runs logged."""
        return self._run_count

    def log_run(self, record: RunRecord) -> None:
        """Append a run record to the JSONL file.

        Args:
            record: The run record to log
        """
        with open(self._runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        self._run_count += 1
        logger.debug(f"Logged run {record.run_number}")

    def log_run_from_dict(self, run_dict: Dict[str, Any]) -> None:
        """Append a run record from a dictionary.

        Args:
            run_dict: Dictionary with run record fields
        """
        with open(self._runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_dict, ensure_ascii=False) + "\n")

        self._run_count += 1

    def read_all_runs(self) -> List[RunRecord]:
        """Read all run records from the file.

        Returns:
            List of RunRecord instances
        """
        if not self._runs_path.exists():
            return []

        records = []
        with open(self._runs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(RunRecord(**data))
        return records

    def read_runs_as_dicts(self) -> List[Dict[str, Any]]:
        """Read all run records as dictionaries.

        Returns:
            List of run record dictionaries
        """
        if not self._runs_path.exists():
            return []

        records = []
        with open(self._runs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
