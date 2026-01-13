# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project for training an agent to control the LunarLanderContinuous-v3 environment from Gymnasium. The project uses:
- **PyTorch** for neural networks (TD3 algorithm)
- **Gymnasium** for the environment
- **Pygame** for rendering
- **Matplotlib** for training visualization

## Project Structure

```
lunar-lander/
├── main.py                # Entry point - training orchestration
├── config.py              # Configuration dataclasses
├── data_types.py          # Shared data structures (Experience, EpisodeResult, etc.)
│
├── training/              # Core RL algorithm components
│   ├── network.py         # Actor/Critic neural networks, OUActionNoise
│   ├── trainer.py         # TD3Trainer class
│   ├── replay_buffer.py   # ReplayBuffer and PrioritizedReplayBuffer
│   └── environment.py     # Gym wrappers, reward shaping, EpisodeManager
│
├── analysis/              # Analysis and visualization
│   ├── behavior_analysis.py  # BehaviorAnalyzer - detects flight patterns
│   ├── diagnostics.py        # DiagnosticsTracker/Reporter - metrics
│   └── charts.py             # ChartGenerator - training visualizations
│
└── docs/                  # Documentation
    ├── CLAUDE.md          # This file
    ├── BEHAVIORS.md       # Behavior detection documentation
    ├── LESSONS_LEARNED.md # Reward shaping lessons
    └── plan.txt           # Task planning
```

## Environment Setup

The project uses Python 3.12.5 with a virtual environment at `.venv-3.12.5/`.

### Activating the Virtual Environment

**Windows:**
```bash
.venv-3.12.5/Scripts/activate
```

**Linux/Mac:**
```bash
source .venv-3.12.5/bin/activate
```

### Installing Dependencies

```bash
python -m pip install torch gymnasium pygame numpy matplotlib box2d
```

## Running the Simulation

```bash
python main.py
```

Key configuration options in `config.py`:
- `training_enabled`: Set to `False` to skip training (fast simulation for testing)
- `render_mode`: `'all'`, `'none'`, or `'custom'` for rendering episodes
- `num_episodes`: Total episodes to train
- `num_envs`: Number of parallel environments

## Code Architecture

### Training Flow

1. `main.py` creates environments via `training/environment.py`
2. For rendered episodes: single environment with Pygame display
3. For non-rendered: `AsyncVectorEnv` with N parallel environments
4. Actions come from `training/network.py` Actor + exploration noise
5. Experiences stored in `training/replay_buffer.py`
6. `training/trainer.py` performs TD3 updates (critics, delayed actor, soft target updates)
7. `analysis/diagnostics.py` tracks metrics throughout
8. `analysis/charts.py` generates visualizations at end

### Key Components

- **TD3Trainer**: Twin Delayed DDPG with clipped double Q-learning
- **PrioritizedReplayBuffer**: Sum tree for O(log n) priority sampling
- **BehaviorAnalyzer**: Detects 50+ flight behaviors for diagnostics
- **EpisodeManager**: Pre-allocated arrays for efficient step tracking

### Reward Shaping (training/environment.py)

Custom shaping on top of Gymnasium rewards:
- Time penalty: -0.05/step (discourages hovering)
- Low altitude bonus (if descending)
- Leg contact bonus (if descending)
- Stability bonus (if descending)
- Terminal landing bonus: +100

See `docs/LESSONS_LEARNED.md` for reward shaping pitfalls.

## Development Notes

### Performance Optimizations

The codebase includes several optimizations:
- Iterative (not recursive) SumTree propagation
- Vectorized numpy operations in replay buffer
- Pre-allocated arrays in EpisodeManager
- Batched tensor detach/numpy conversions
- Single-pass behavior statistics computation

### Important Files

- **docs/LESSONS_LEARNED.md**: Critical lessons about reward shaping mistakes
- **docs/BEHAVIORS.md**: Full documentation of detected behaviors
- **docs/plan.txt**: Current task planning

## Troubleshooting

### Virtual Environment Issues

Use `.venv-3.12.5` which contains all required packages.

### Pygame Warnings

Deprecation warnings about `pkg_resources` are expected and can be ignored.
