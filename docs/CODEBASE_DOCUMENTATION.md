# Lunar Lander TD3 Codebase Documentation

## Overview

This is a **TD3 (Twin Delayed DDPG) reinforcement learning** implementation for the Gymnasium LunarLanderContinuous-v3 environment. The project trains an agent to land a spacecraft on a landing pad using continuous throttle control.

**Key Technologies:**
- PyTorch for neural networks
- Gymnasium for the environment
- Vectorized environments (AsyncVectorEnv) for parallel training
- Prioritized Experience Replay (PER) with Sum Tree

---

## Directory Structure

```
lunar-lander/
├── src/                           # Main source code
│   ├── main.py                    # Entry point - training loop
│   ├── config.py                  # All configuration dataclasses
│   ├── data_types.py              # Data structures (Experience, Batch, Metrics)
│   ├── constants.py               # Behavior constants and mappings
│   ├── models.py                  # Re-exports for convenience
│   │
│   ├── training/                  # Training components
│   │   ├── trainer.py             # TD3Trainer class
│   │   ├── network.py             # ActorNetwork, CriticNetwork
│   │   ├── replay_buffer.py       # ReplayBuffer, PrioritizedReplayBuffer, SumTree
│   │   ├── environment.py         # Environment management, reward shaping
│   │   └── noise.py               # OUActionNoise (Ornstein-Uhlenbeck)
│   │
│   └── analysis/                  # Analysis and diagnostics
│       ├── behavior_analysis.py   # BehaviorAnalyzer - detects landing patterns
│       ├── diagnostics.py         # DiagnosticsTracker - metrics collection
│       └── reporter.py            # DiagnosticsReporter - generates reports
│
├── tools/                         # CLI tools
│   ├── sweep_runner.py            # Hyperparameter sweep execution
│   ├── profiler.py                # Performance profiling
│   ├── compare_runs.py            # Compare training logs
│   └── log_analyzer.py            # Parse training output
│
├── tests/                         # Pytest unit tests
│   ├── conftest.py                # Shared fixtures
│   ├── test_config.py
│   ├── test_replay_buffer.py
│   ├── test_behavior_analysis.py
│   ├── test_trainer.py
│   └── test_environment.py
│
├── sweep_configs/                 # Hyperparameter sweep configurations
│   ├── example_lr_sweep.json      # Grid search over learning rates
│   ├── example_random_sweep.json  # Random search example
│   └── example_noise_sweep.json   # Noise parameter sweep
│
├── docs/                          # Training logs and documentation
├── chart_images/                  # Generated training charts
└── pyproject.toml                 # Project config (pytest, black, ruff, mypy)
```

---

## Core Components

### 1. Configuration System (`src/config.py`)

All hyperparameters are organized into **frozen dataclasses** for immutability:

```python
@dataclass(frozen=True)
class TrainingConfig:
    actor_lr: float = 0.001          # Actor learning rate
    critic_lr: float = 0.002         # Critic learning rate
    gamma: float = 0.99              # Discount factor
    tau: float = 0.005               # Soft update rate for target networks
    batch_size: int = 128            # Training batch size
    buffer_size: int = 16384         # Replay buffer capacity
    min_experiences_before_training: int = 512
    policy_update_frequency: int = 3 # TD3 delayed policy updates
    training_updates_per_episode: int = 25

    # Prioritized Experience Replay
    use_per: bool = True
    per_alpha: float = 0.6           # Priority exponent
    per_beta_start: float = 0.4      # Initial importance sampling weight
    per_beta_end: float = 1.0        # Final importance sampling weight
    per_epsilon: float = 1e-6

    # TD3 specific
    target_policy_noise: float = 0.2 # Noise added to target actions
    target_noise_clip: float = 0.5   # Noise clipping range
    gradient_clip_value: float = 1.0

@dataclass(frozen=True)
class NoiseConfig:
    mu: tuple = (0.0, 0.0)           # OU noise mean
    sigma: float = 0.3               # OU noise standard deviation
    theta: float = 0.2               # OU noise mean reversion rate
    dt: float = 0.01                 # OU noise time step
    noise_scale_initial: float = 1.0 # Starting exploration noise multiplier
    noise_scale_final: float = 0.2   # Final exploration noise multiplier
    noise_decay_episodes: int = 300  # Episodes to decay noise

@dataclass(frozen=True)
class RunConfig:
    num_episodes: int = 5000
    num_envs: int = 8                # Parallel environments
    render_mode: str = 'all'         # 'all', 'none', or 'custom'
    training_enabled: bool = True
    random_warmup_episodes: int = 30 # Random actions before training

@dataclass(frozen=True)
class EnvironmentConfig:
    env_name: str = "LunarLanderContinuous-v3"
    state_dim: int = 8
    action_dim: int = 2
    success_threshold: float = 200.0 # Reward >= 200 = success
```

**Key Point:** The master `Config` class combines all sub-configs:
```python
class Config:
    training: TrainingConfig
    noise: NoiseConfig
    run: RunConfig
    environment: EnvironmentConfig
    display: DisplayConfig
```

---

### 2. Neural Networks (`src/training/network.py`)

**ActorNetwork** - Maps states to actions:
- Architecture: `state(8) -> 256 -> LayerNorm -> 128 -> LayerNorm -> action(2)`
- Separate output heads for main engine and side engine
- `tanh` activation for bounded actions [-1, 1]
- Main engine bias initialized to 0.3 (encourages initial thrust)

**CriticNetwork** - Estimates Q-values:
- Architecture: `state(8) -> 256 -> LayerNorm -> concat(features, action) -> 128 -> LayerNorm -> Q(1)`
- TD3 uses TWO critics (take minimum to reduce overestimation)

**Utility functions:**
- `soft_update(source, target, tau)` - Polyak averaging for target networks
- `hard_update(source, target)` - Direct weight copy

---

### 3. TD3 Trainer (`src/training/trainer.py`)

The `TD3Trainer` class implements the TD3 algorithm:

```python
class TD3Trainer:
    def __init__(training_config, env_config, run_config, device):
        # Creates: actor, critic_1, critic_2
        # Creates: target_actor, target_critic_1, target_critic_2
        # Creates: optimizers with weight decay
        # Creates: ExponentialLR schedulers (decay to 20% over training)

    def train_step(batch: ExperienceBatch) -> (TrainingMetrics, td_errors):
        # 1. Compute target Q-values using target networks
        # 2. Add noise to target actions (target policy smoothing)
        # 3. Take minimum of two critic Q-values (clipped double Q)
        # 4. Update both critics with smooth L1 loss
        # 5. Every `policy_update_frequency` steps:
        #    - Update actor to maximize Q
        #    - Soft update all target networks
        # Returns: metrics and TD errors (for PER priority updates)

    def train_on_buffer(replay_buffer, num_updates) -> AggregatedTrainingMetrics:
        # Performs `num_updates` training steps from buffer
```

**TD3 Key Features:**
1. **Clipped Double Q-Learning**: Two critics, use minimum Q to compute targets
2. **Delayed Policy Updates**: Update actor every `policy_update_frequency` steps (default: 3)
3. **Target Policy Smoothing**: Add noise to target actions during critic update

---

### 4. Replay Buffers (`src/training/replay_buffer.py`)

**ReplayBuffer** - Uniform sampling:
```python
class ReplayBuffer:
    def push(experience: Experience)  # O(1) circular buffer
    def sample(batch_size) -> ExperienceBatch  # Random sampling
    def is_ready(min_size) -> bool
```

**PrioritizedReplayBuffer** - Priority-based sampling using SumTree:
```python
class PrioritizedReplayBuffer:
    def push(experience)              # New experiences get max priority
    def sample(batch_size) -> ExperienceBatch  # Includes weights and indices
    def update_priorities(indices, priorities)  # Update after training
    def anneal_beta(progress)         # Increase importance sampling weight
```

**SumTree** - O(log n) priority-based sampling:
- Binary tree where parents store sum of children
- Enables proportional sampling by cumulative sum lookup

---

### 5. Environment Management (`src/training/environment.py`)

**`create_environments(run_config, env_config)`** - Context manager:
- Creates `AsyncVectorEnv` with `num_envs` parallel environments
- Creates separate render environment for visualization

**`shape_reward(state, base_reward, terminated, step)`** - Reward shaping:
```python
# Shaping rewards (gated on descending to prevent hover exploitation):
shaped_reward = base_reward
shaped_reward -= 0.05  # Time penalty (discourages hovering)

if is_descending (y_vel < -0.05):
    if y_pos < 0.25: shaped_reward += 0.5      # Close to ground
    if one_leg_contact: shaped_reward += 2     # Single leg
    if both_legs_contact: shaped_reward += 5   # Both legs
    if abs(angle) < 0.1: shaped_reward += 0.3  # Upright
    elif abs(angle) < 0.2: shaped_reward += 0.1

if terminated and both_legs and base_reward > 0:
    shaped_reward += 100  # Terminal landing bonus
```

**`compute_noise_scale(episode, initial, final, decay_episodes)`** - Linear decay

**`EpisodeManager`** - Tracks per-environment episode state:
- Pre-allocates arrays for rewards, actions, observations
- Tracks shaped bonuses incrementally

---

### 6. Exploration Noise (`src/training/noise.py`)

**OUActionNoise** - Ornstein-Uhlenbeck process:
- Temporally correlated noise (beneficial for inertial control)
- Formula: `noise += theta * (mu - noise) * dt + sigma * sqrt(dt) * random`
- Supports multiple environments with independent noise states

---

### 7. Behavior Analysis (`src/analysis/behavior_analysis.py`)

**BehaviorAnalyzer** - Analyzes episode trajectories to detect patterns:

```python
def analyze(observations, actions, terminated, truncated) -> BehaviorReport:
    # Returns: outcome (str) + behaviors (List[str])
```

**Outcomes** (mutually exclusive):
- `LANDED_PERFECTLY`, `LANDED_SOFTLY`, `LANDED_HARD`, `LANDED_TILTED`
- `CRASHED_HIGH_VELOCITY`, `CRASHED_SPINNING`, `CRASHED_SIDEWAYS`, `CRASHED_TILTED`
- `FLEW_OFF_LEFT`, `FLEW_OFF_RIGHT`, `FLEW_OFF_TOP`
- `TIMED_OUT_HOVERING`, `TIMED_OUT_DESCENDING`, `TIMED_OUT_ASCENDING`

**Behavior Patterns Detected:**
- Vertical: `CONTROLLED_DESCENT`, `RAPID_DESCENT`, `FREEFALL`, `YO_YO_PATTERN`
- Horizontal: `STAYED_CENTERED`, `DRIFTED_LEFT/RIGHT`, `HORIZONTAL_OSCILLATION`
- Orientation: `STAYED_UPRIGHT`, `HEAVY_LEFT/RIGHT_TILT`, `SPINNING_UNCONTROLLED`
- Thrust: `MAIN_THRUST_HEAVY/MODERATE/LIGHT/NONE`, `ERRATIC_THRUST`, `SMOOTH_THRUST`
- Contact: `TOUCHED_DOWN_CLEAN`, `BOUNCED`, `SCRAPED_LEFT/RIGHT_LEG`
- Quality: `CONTROLLED_THROUGHOUT`, `NEVER_STABILIZED`, `LOST_CONTROL_LATE`

---

### 8. Diagnostics Tracking (`src/analysis/diagnostics.py`)

**DiagnosticsTracker** - Incremental metrics collection:
- Uses running sums and counters (O(1) per episode, not O(n) recomputation)
- Memory efficient: ~78 bytes/episode vs ~368 bytes with objects
- Tracks per-batch statistics automatically

Key methods:
```python
def record_episode(episode_num, env_reward, shaped_bonus, duration, success)
def record_behavior(outcome, behaviors, env_reward, success)
def record_training_metrics(aggregated_metrics)
def get_summary() -> DiagnosticsSummary
def get_behavior_statistics() -> BehaviorStatistics
```

---

### 9. Data Types (`src/data_types.py`)

```python
@dataclass
class Experience:
    state: T.Tensor        # Shape: (8,)
    action: T.Tensor       # Shape: (2,)
    reward: T.Tensor       # Scalar
    next_state: T.Tensor   # Shape: (8,)
    done: T.Tensor         # Boolean

@dataclass
class ExperienceBatch:
    states: T.Tensor       # Shape: (batch, 8)
    actions: T.Tensor      # Shape: (batch, 2)
    rewards: T.Tensor      # Shape: (batch,)
    next_states: T.Tensor  # Shape: (batch, 8)
    dones: T.Tensor        # Shape: (batch,)
    weights: Optional[T.Tensor]   # For PER importance sampling
    indices: Optional[np.ndarray] # For PER priority updates

@dataclass
class TrainingMetrics:
    critic_loss: float
    actor_loss: float
    avg_q_value: float
    critic_grad_norm: float
    actor_grad_norm: float
```

---

## Hyperparameter Sweep Infrastructure

### Sweep Runner (`tools/sweep_runner.py`)

**Capabilities:**
1. **Grid Search**: Test all combinations of parameter values
2. **Random Search**: Sample from parameter ranges (supports log-uniform for learning rates)

**Sweep Config Format:**
```json
{
  "name": "lr_sweep",
  "type": "grid",              // or "random"
  "episodes_per_run": 500,
  "num_samples": 10,           // for random search
  "seed": 42,
  "parameters": {
    // Grid search: list of values
    "actor_lr": [0.0005, 0.001, 0.002],
    "critic_lr": [0.001, 0.002, 0.004],

    // Random search: range specification
    "actor_lr": {"min": 0.0001, "max": 0.01, "type": "log"},
    "batch_size": {"min": 32, "max": 256, "type": "int"},
    "tau": {"min": 0.001, "max": 0.01, "type": "float"}
  }
}
```

**Key Functions:**
```python
def load_sweep_config(config_path) -> Dict
def generate_grid_configs(base_config, parameters) -> Iterator[(params, Config)]
def generate_random_configs(base_config, parameters, num_samples, seed) -> Iterator
def apply_params_to_config(config, params) -> Config  # Creates modified Config
def run_training_with_config(config, run_name, results_dir) -> Dict  # Full training
def run_sweep(sweep_config, dry_run) -> List[Dict]  # Execute sweep
def generate_sweep_summary(results, results_dir)  # CSV + summary
```

**Output:**
- `sweep_results/[timestamp]_[name]/`
  - `sweep_config.json` - Original config
  - `run_XXX_results.json` - Per-run results
  - `summary.csv` - Comparison table
  - `all_results.json` - Complete results

---

## Key Hyperparameters for Tuning

### Most Impactful (Recommended for Sweeps)

| Parameter | Config | Default | Typical Range | Notes |
|-----------|--------|---------|---------------|-------|
| `actor_lr` | TrainingConfig | 0.001 | 0.0001 - 0.01 | Use log-uniform sampling |
| `critic_lr` | TrainingConfig | 0.002 | 0.0001 - 0.01 | Usually 2x actor_lr |
| `batch_size` | TrainingConfig | 128 | 32 - 256 | Larger = more stable |
| `tau` | TrainingConfig | 0.005 | 0.001 - 0.02 | Target update rate |
| `sigma` | NoiseConfig | 0.3 | 0.1 - 0.5 | Exploration noise |

### Secondary Parameters

| Parameter | Config | Default | Notes |
|-----------|--------|---------|-------|
| `gamma` | TrainingConfig | 0.99 | Discount factor (rarely changed) |
| `buffer_size` | TrainingConfig | 16384 | Memory/speed tradeoff |
| `policy_update_frequency` | TrainingConfig | 3 | TD3 delayed updates |
| `noise_scale_initial` | NoiseConfig | 1.0 | Starting exploration |
| `noise_scale_final` | NoiseConfig | 0.2 | Final exploration |
| `noise_decay_episodes` | NoiseConfig | 300 | Exploration decay schedule |
| `per_alpha` | TrainingConfig | 0.6 | PER priority exponent |
| `per_beta_start` | TrainingConfig | 0.4 | PER importance sampling |

### Network Architecture (requires code changes)

Current: `256 -> 128` (reduced from typical `400 -> 300`)
- `hidden1` in ActorNetwork/CriticNetwork: 256
- `hidden2` in ActorNetwork/CriticNetwork: 128

---

## Environment Details

**LunarLanderContinuous-v3:**
- **State Space**: 8-dimensional continuous
  - `[x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]`
  - Positions normalized to ~[-1, 1], velocities can exceed
  - `leg_contact` is binary (0 or 1)

- **Action Space**: 2-dimensional continuous [-1, 1]
  - `action[0]`: Main engine thrust (-1 = off, 1 = full thrust)
  - `action[1]`: Side engine thrust (-1 = left, 1 = right)

- **Rewards**:
  - Moving toward pad: +100 to +140
  - Moving away: -100 to -140
  - Crash: -100
  - Successful landing: +100 to +140 (base) + leg bonuses
  - Each leg contact: +10
  - Firing main engine: -0.3 per frame
  - Firing side engine: -0.03 per frame

- **Termination**:
  - Crash (contact outside legs, high velocity, or out of bounds)
  - Truncation at 1000 steps

---

## Running the Code

### Training
```bash
cd lunar-lander
python src/main.py
```

### Hyperparameter Sweep
```bash
# Dry run (preview configurations)
python tools/sweep_runner.py sweep_configs/example_lr_sweep.json --dry-run

# Execute sweep
python tools/sweep_runner.py sweep_configs/example_lr_sweep.json
```

### Analyze Training Log
```bash
python tools/log_analyzer.py docs/training_output.txt
```

### Compare Runs
```bash
python tools/compare_runs.py docs/
```

### Run Tests
```bash
pytest tests/ -v
```

### Linting
```bash
black src/
ruff check src/
mypy src/
```

---

## Implementing New Hyperparameter Testing Infrastructure

### To Add a New Parameter to Sweeps

1. **Identify the parameter** in `config.py` (TrainingConfig, NoiseConfig, or RunConfig)
2. **Add to sweep config JSON** under `parameters`
3. The `apply_params_to_config()` function automatically routes parameters to the correct sub-config

### To Add a New Metric to Track

1. **Add to `DiagnosticsTracker`** in `diagnostics.py`
2. **Update `record_episode()` or `record_behavior()`** to capture the data
3. **Add getter method** for charts/analysis
4. **Update `sweep_runner.py`** `run_training_with_config()` to include in results

### To Modify Training Loop for Sweeps

The `run_training_with_config()` function in `tools/sweep_runner.py` contains a complete training loop that:
- Creates trainer, buffer, noise
- Runs episodes with vectorized environments
- Handles experience collection and training
- Returns results dictionary

### Architecture for Automated Hyperparameter Testing

Recommended approach:
1. **Sweep orchestrator**: Use existing `sweep_runner.py` or extend it
2. **Configuration**: JSON configs in `sweep_configs/`
3. **Results storage**: `sweep_results/[timestamp]/` with JSON + CSV
4. **Analysis**: Use `compare_runs.py` or extend with visualization
5. **Bayesian optimization**: Consider integrating Optuna for smarter search

---

## Key Implementation Notes

1. **Vectorized Environments**: Training uses `AsyncVectorEnv` with 8 parallel environments by default. Each step returns batched observations/rewards for all environments.

2. **Reward Shaping Gating**: All shaping bonuses are gated on `is_descending` to prevent hover exploitation where the agent learns to hover and farm contact bonuses.

3. **PER Integration**: The trainer returns TD errors which are used to update priorities in `PrioritizedReplayBuffer`.

4. **LR Scheduling**: Learning rates decay to 20% of initial value over training using `ExponentialLR`.

5. **Frozen Configs**: All config dataclasses are frozen to prevent accidental modification. Use `dataclasses.replace()` to create modified copies.

6. **Incremental Statistics**: `DiagnosticsTracker` uses running sums and counters to avoid O(n) recomputation.
