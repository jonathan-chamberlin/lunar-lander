---
description: Generate pytest unit tests for project components
---

Generate unit tests for the lunar lander TD3 project.

## Target Test Areas

1. **Behavior Analysis** (`analysis/behavior_analysis.py`)
   - Test `BehaviorAnalyzer.analyze()` with mock observations/actions
   - Test threshold detection for each behavior category
   - Test outcome classification (LANDED_PERFECTLY, CRASHED_HIGH_VELOCITY, etc.)

2. **Replay Buffer** (`training/replay_buffer.py`)
   - Test `ReplayBuffer.push()` and circular overwrite
   - Test `ReplayBuffer.sample()` returns correct batch shape
   - Test `PrioritizedReplayBuffer` priority updates
   - Test `SumTree` operations

3. **TD3 Trainer** (`training/trainer.py`)
   - Test network initialization
   - Test `train_on_buffer()` runs without error
   - Test target network soft updates

4. **Configuration** (`config.py`)
   - Test frozen dataclass behavior (immutability)
   - Test `RunConfig.__post_init__` sets render_episodes correctly
   - Test default values

5. **Environment** (`training/environment.py`)
   - Test `shape_reward()` function
   - Test `compute_noise_scale()` decay

## Steps

1. Check if `tests/` directory exists, create if not
2. Check if `pyproject.toml` has pytest config, add if not
3. Read source files to understand function signatures
4. Generate test files with:
   - Fixtures in `conftest.py`
   - Deterministic tests (set random seeds)
   - Edge cases and boundary conditions
   - Mock objects where needed (e.g., mock environments)

## Test File Structure

```
lunar-lander/
  tests/
    __init__.py
    conftest.py           # Shared fixtures
    test_behavior_analysis.py
    test_replay_buffer.py
    test_trainer.py
    test_config.py
    test_environment.py
```

## Running Tests

```bash
cd lunar-lander
pytest tests/ -v
```

Or with coverage:
```bash
pytest tests/ --cov=. --cov-report=term-missing
```
