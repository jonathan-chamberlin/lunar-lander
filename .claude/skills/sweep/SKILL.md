---
name: sweep
description: Run hyperparameter sweep experiments for RL training. Use when user wants to test multiple hyperparameter configurations, find optimal settings, tune learning rates, or compare different training setups systematically.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Hyperparameter Sweep

Help the user set up and run hyperparameter sweeps for the lunar lander TD3 training.

## When to Use This Skill

- User wants to find optimal hyperparameters
- User wants to run an expiriment.
- User mentions "sweep", "grid search", "hyperparameter tuning"
- User wants to compare multiple configurations systematically
- User wants to improve how fast the agent learns and reaches a successful stable state

## Available Hyperparameters

From `config.py` TrainingConfig:
- `actor_lr` - Actor learning rate (default: 0.001)
- `critic_lr` - Critic learning rate (default: 0.002)
- `batch_size` - Training batch size (default: 128)
- `buffer_size` - Replay buffer capacity (default: 16384)
- `gamma` - Discount factor (default: 0.99)
- `tau` - Target network soft update rate (default: 0.005)
- `policy_update_frequency` - TD3 delayed policy updates (default: 3)
- `training_updates_per_episode` - Updates per episode (default: 25)

From `config.py` NoiseConfig:
- `sigma` - OU noise sigma (default: 0.3)
- `theta` - OU noise theta (default: 0.2)
- `noise_scale_initial` - Starting noise multiplier (default: 1.0)
- `noise_scale_final` - Final noise multiplier (default: 0.2)
- `noise_decay_episodes` - Episodes to decay noise (default: 300)

## Workflow

1. **Generate config**: Run `python scripts/generate_config.py` to create a sweep configuration
2. **Execute sweep**: Run `python scripts/sweep_runner.py config.json` to execute
3. **Analyze results**: Run `python scripts/analyze_results.py results_dir/` to summarize

## Reference Documentation

- For recommended parameter ranges, see [hyperparameter_ranges.md](references/hyperparameter_ranges.md)
- For sweep strategy guidance, see [sweep_strategies.md](references/sweep_strategies.md)

## Example Sweep Config

```json
{
  "name": "lr_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "parameters": {
    "actor_lr": [0.0005, 0.001, 0.002],
    "critic_lr": [0.001, 0.002, 0.004]
  }
}
```

## Output

Results saved to `sweep_results/[timestamp]/` with:
- Individual run logs
- Summary CSV with all configurations and metrics
- Best configuration recommendation
