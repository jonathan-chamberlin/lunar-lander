"""Integration tests for the data pipeline.

Tests the data architecture including:
- Simulation directory creation and isolation
- Config write-once semantics
- Run logger append-only behavior
- Aggregate writer periodic snapshots
- Experiment aggregation
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.abstractions import SimulationIdentifier, RunIdentifier, ExperimentDefinition
from data.simulation_io import (
    SimulationDirectory,
    ConfigAlreadyWrittenError,
    IsolationViolationError,
)
from data.config_serializer import serialize_config, create_config_snapshot
from data.run_logger import RunLogger, RunRecord
from data.aggregate_writer import AggregateWriter
from data.experiment_io import ExperimentDirectory, ExperimentExistsError


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


class TestSimulationIdentifier:
    """Tests for SimulationIdentifier."""

    def test_create_unique(self):
        """Each creation should produce a unique identifier."""
        id1 = SimulationIdentifier.create()
        id2 = SimulationIdentifier.create()
        assert str(id1) != str(id2)

    def test_parse_roundtrip(self):
        """Parsing a string representation should recover the original."""
        original = SimulationIdentifier.create()
        parsed = SimulationIdentifier.from_string(str(original))
        assert parsed.timestamp == original.timestamp
        assert parsed.uuid_short == original.uuid_short

    def test_directory_name(self):
        """Directory name should match string representation."""
        sim_id = SimulationIdentifier.create()
        assert sim_id.directory_name == str(sim_id)


class TestRunIdentifier:
    """Tests for RunIdentifier."""

    def test_create_valid(self):
        """Creating a valid run identifier should work."""
        sim_id = SimulationIdentifier.create()
        run_id = RunIdentifier(sim_id, 0)
        assert run_id.run_number == 0

    def test_negative_run_number_fails(self):
        """Negative run numbers should raise ValueError."""
        sim_id = SimulationIdentifier.create()
        with pytest.raises(ValueError):
            RunIdentifier(sim_id, -1)


class TestExperimentDefinition:
    """Tests for ExperimentDefinition."""

    def test_invalid_name_fails(self):
        """Invalid characters in name should raise ValueError."""
        with pytest.raises(ValueError):
            ExperimentDefinition(name="test/invalid")

    def test_empty_name_fails(self):
        """Empty name should raise ValueError."""
        with pytest.raises(ValueError):
            ExperimentDefinition(name="   ")

    def test_add_simulation(self):
        """Adding simulations should work."""
        exp = ExperimentDefinition(name="test")
        sim_id = SimulationIdentifier.create()
        exp.add_simulation(sim_id)
        assert str(sim_id) in exp.simulation_ids

    def test_add_duplicate_simulation(self):
        """Adding the same simulation twice should not duplicate."""
        exp = ExperimentDefinition(name="test")
        sim_id = SimulationIdentifier.create()
        exp.add_simulation(sim_id)
        exp.add_simulation(sim_id)
        assert exp.simulation_count == 1


class TestSimulationDirectory:
    """Tests for SimulationDirectory."""

    def test_initialize_creates_structure(self, temp_dir):
        """Initialize should create the expected directory structure."""
        sim_dir = SimulationDirectory(temp_dir)
        sim_dir.initialize()

        assert sim_dir.root_path.exists()
        assert sim_dir.aggregates_path.exists()
        assert sim_dir.models_path.exists()
        assert sim_dir.charts_path.exists()
        assert sim_dir.text_path.exists()

    def test_isolation_violation(self, temp_dir):
        """Initializing twice with same ID should raise error."""
        sim_id = SimulationIdentifier.create()

        sim_dir1 = SimulationDirectory(temp_dir, sim_id)
        sim_dir1.initialize()

        sim_dir2 = SimulationDirectory(temp_dir, sim_id)
        with pytest.raises(IsolationViolationError):
            sim_dir2.initialize()

    def test_config_write_once(self, temp_dir):
        """Writing config twice should raise error."""
        sim_dir = SimulationDirectory(temp_dir)
        sim_dir.initialize()

        config = {"key": "value"}
        sim_dir.write_config(config)

        with pytest.raises(ConfigAlreadyWrittenError):
            sim_dir.write_config(config)

    def test_config_roundtrip(self, temp_dir):
        """Config should be readable after writing."""
        sim_dir = SimulationDirectory(temp_dir)
        sim_dir.initialize()

        config = {"key": "value", "nested": {"a": 1}}
        sim_dir.write_config(config)

        loaded = sim_dir.read_config()
        assert loaded == config

    def test_append_run(self, temp_dir):
        """Runs should be appendable and readable."""
        sim_dir = SimulationDirectory(temp_dir)
        sim_dir.initialize()

        sim_dir.append_run({"run": 0, "reward": 100})
        sim_dir.append_run({"run": 1, "reward": 150})

        runs = sim_dir.read_runs()
        assert len(runs) == 2
        assert runs[0]["run"] == 0
        assert runs[1]["run"] == 1


class TestConfigSerializer:
    """Tests for config serialization."""

    def test_serialize_dataclass(self):
        """Dataclasses should be serializable."""
        from dataclasses import dataclass

        @dataclass
        class TestConfig:
            value: int = 42
            name: str = "test"

        config = TestConfig()
        serialized = serialize_config(config)

        assert serialized["value"] == 42
        assert serialized["name"] == "test"

    def test_serialize_nested(self):
        """Nested dataclasses should be serializable."""
        from dataclasses import dataclass

        @dataclass
        class Inner:
            x: int = 1

        @dataclass
        class Outer:
            inner: Inner = None

            def __post_init__(self):
                if self.inner is None:
                    self.inner = Inner()

        config = Outer()
        serialized = serialize_config(config)

        assert serialized["inner"]["x"] == 1

    def test_create_snapshot_has_metadata(self):
        """Snapshots should include metadata."""
        from dataclasses import dataclass

        @dataclass
        class TestConfig:
            value: int = 42

        config = TestConfig()
        sim_id = SimulationIdentifier.create()
        snapshot = create_config_snapshot(config, sim_id)

        assert "_metadata" in snapshot
        assert "config" in snapshot
        assert snapshot["_metadata"]["simulation_id"] == str(sim_id)


class TestRunLogger:
    """Tests for RunLogger."""

    def test_log_run(self, temp_dir):
        """Logging runs should create JSONL entries."""
        runs_path = Path(temp_dir) / "runs.jsonl"
        logger = RunLogger(runs_path)

        record = RunRecord.create(
            run_number=0,
            env_reward=100.0,
            shaped_bonus=10.0,
            steps=200,
            duration_seconds=1.5,
            success=True,
            outcome="LANDED_SAFE",
            behaviors=["STAYED_UPRIGHT", "CONTROLLED_DESCENT"],
            terminated=True,
            truncated=False,
            rendered=False,
        )
        logger.log_run(record)

        assert runs_path.exists()
        runs = logger.read_all_runs()
        assert len(runs) == 1
        assert runs[0].run_number == 0
        assert runs[0].env_reward == 100.0

    def test_append_only(self, temp_dir):
        """Multiple logs should append, not overwrite."""
        runs_path = Path(temp_dir) / "runs.jsonl"
        logger = RunLogger(runs_path)

        for i in range(3):
            record = RunRecord.create(
                run_number=i,
                env_reward=100.0 + i,
                shaped_bonus=0.0,
                steps=100,
                duration_seconds=1.0,
                success=False,
                outcome="CRASHED",
                behaviors=[],
                terminated=True,
                truncated=False,
                rendered=False,
            )
            logger.log_run(record)

        runs = logger.read_all_runs()
        assert len(runs) == 3
        assert runs[2].run_number == 2


class TestAggregateWriter:
    """Tests for AggregateWriter."""

    def test_maybe_write_interval(self, temp_dir):
        """Writer should write at specified intervals."""
        agg_path = Path(temp_dir)
        agg_path.mkdir(exist_ok=True)
        writer = AggregateWriter(agg_path, write_interval=100)

        # Create a mock diagnostics tracker
        class MockDiagnostics:
            def get_summary(self):
                from dataclasses import dataclass

                @dataclass
                class Summary:
                    total_episodes: int = 100
                    num_successes: int = 50
                    success_rate: float = 0.5
                    mean_reward: float = 150.0
                    max_reward: float = 250.0
                    min_reward: float = 50.0
                    final_50_mean_reward: float = 175.0
                    mean_q_value: float = None
                    q_value_trend: tuple = None
                    mean_actor_loss: float = None
                    mean_critic_loss: float = None
                    mean_actor_grad_norm: float = None
                    mean_critic_grad_norm: float = None

                return Summary()

            def get_behavior_statistics(self):
                return None

            def get_advanced_statistics(self):
                return {}

            def get_streak_statistics(self):
                return {"max_streak": 5, "max_streak_episode": 50, "current_streak": 0}

            def get_env_reward_distribution(self):
                return {}

        diagnostics = MockDiagnostics()

        # Should not write at 50
        assert not writer.maybe_write(50, diagnostics)

        # Should write at 100
        assert writer.maybe_write(100, diagnostics)
        assert (agg_path / "batch_0001.json").exists()

        # Should not write at 100 again
        assert not writer.maybe_write(100, diagnostics)

        # Should write at 200
        assert writer.maybe_write(200, diagnostics)
        assert (agg_path / "batch_0002.json").exists()


class TestExperimentDirectory:
    """Tests for ExperimentDirectory."""

    def test_initialize(self, temp_dir):
        """Initialize should create experiment structure."""
        exp_dir = ExperimentDirectory(temp_dir, "test_exp")
        exp_dir.initialize(description="Test experiment")

        assert exp_dir.experiment_path.exists()

    def test_experiment_exists_error(self, temp_dir):
        """Initializing existing experiment should raise error."""
        exp_dir = ExperimentDirectory(temp_dir, "test_exp")
        exp_dir.initialize()

        exp_dir2 = ExperimentDirectory(temp_dir, "test_exp")
        with pytest.raises(ExperimentExistsError):
            exp_dir2.initialize()

    def test_allow_existing(self, temp_dir):
        """Allow existing should load instead of error."""
        exp_dir = ExperimentDirectory(temp_dir, "test_exp")
        exp_dir.initialize(description="Test")

        exp_dir2 = ExperimentDirectory(temp_dir, "test_exp")
        exp_dir2.initialize(allow_existing=True)
        assert exp_dir2.definition is not None

    def test_add_simulation(self, temp_dir):
        """Adding simulations should persist."""
        exp_dir = ExperimentDirectory(temp_dir, "test_exp")
        exp_dir.initialize()

        sim_id = SimulationIdentifier.create()
        exp_dir.add_simulation(sim_id)

        # Reload and check
        exp_dir2 = ExperimentDirectory(temp_dir, "test_exp")
        exp_dir2.load()
        assert str(sim_id) in exp_dir2.get_simulation_ids()

    def test_list_experiments(self, temp_dir):
        """List experiments should return all experiment names."""
        for name in ["exp1", "exp2", "exp3"]:
            exp_dir = ExperimentDirectory(temp_dir, name)
            exp_dir.initialize()

        experiments = ExperimentDirectory.list_experiments(temp_dir)
        assert set(experiments) == {"exp1", "exp2", "exp3"}


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_simulation_workflow(self, temp_dir):
        """Test complete simulation data flow."""
        # Create simulation directory
        sim_id = SimulationIdentifier.create()
        sim_dir = SimulationDirectory(temp_dir, sim_id)
        sim_dir.initialize()

        # Write config
        config = {"learning_rate": 0.001, "batch_size": 64}
        snapshot = create_config_snapshot(config, sim_id)
        sim_dir.write_config(snapshot)

        # Log runs
        run_logger = RunLogger(sim_dir.runs_path)
        for i in range(10):
            record = RunRecord.create(
                run_number=i,
                env_reward=100.0 + i * 10,
                shaped_bonus=5.0,
                steps=200,
                duration_seconds=1.5,
                success=i >= 5,
                outcome="LANDED_SAFE" if i >= 5 else "CRASHED",
                behaviors=["STAYED_UPRIGHT"],
                terminated=True,
                truncated=False,
                rendered=False,
            )
            run_logger.log_run(record)

        # Verify data
        loaded_config = sim_dir.read_config()
        assert loaded_config["config"]["learning_rate"] == 0.001

        runs = run_logger.read_all_runs()
        assert len(runs) == 10
        assert sum(1 for r in runs if r.success) == 5

    def test_experiment_aggregation(self, temp_dir):
        """Test experiment creation and summary computation."""
        # Create two simulations with aggregates
        for i in range(2):
            sim_id = SimulationIdentifier.create()
            sim_dir = SimulationDirectory(temp_dir, sim_id)
            sim_dir.initialize()

            # Write a mock final aggregate
            aggregate = {
                "summary": {
                    "total_episodes": 100,
                    "success_rate": 0.3 + i * 0.2,
                    "mean_reward": 150 + i * 50,
                    "max_reward": 250,
                },
                "streaks": {
                    "max_streak": 5 + i,
                },
            }
            sim_dir.write_aggregate("final.json", aggregate)

        # Create experiment
        exp_dir = ExperimentDirectory(temp_dir, "comparison")
        exp_dir.initialize(description="Compare two runs")

        # Add simulations
        sims_path = Path(temp_dir) / "simulations"
        for sim_path in sims_path.iterdir():
            sim_id = SimulationIdentifier.from_string(sim_path.name)
            exp_dir.add_simulation(sim_id)

        # Compute summary
        summary = exp_dir.compute_summary(sims_path)

        # Verify summary
        assert summary["_metadata"]["simulation_count"] == 2
        assert "aggregate_metrics" in summary
        assert "best_simulation" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
