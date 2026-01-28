---
name: profile
description: Profile code performance and identify bottlenecks. Use when user wants to find what's slow, measure performance, identify bottlenecks, or understand where time is being spent in their code.
allowed-tools: Read, Write, Bash, Glob, Grep
---

# Profile: Performance Profiling and Bottleneck Detection

Profile Python code to find performance bottlenecks and optimization opportunities.

## When to Use This Skill

- User asks "what's slow" or "where is time being spent"
- User wants to "profile" or "benchmark" code
- User mentions "bottleneck" or "performance issue"
- Before optimization work to identify targets
- User mentions wanting to speed up the simulation.

## Profiling Workflow

1. **CPU Profiling**: Run `python scripts/profiler.py` to identify slow functions
2. **Memory Profiling**: Run `python scripts/memory_profiler.py` to track memory usage
3. **Analyze Results**: Identify top bottlenecks and suggest optimizations

## Reference Documentation

- For interpreting profile output, see [reading_profiles.md](references/reading_profiles.md)
- For common RL bottlenecks, see [common_bottlenecks.md](references/common_bottlenecks.md)

## Quick Start

```bash
# Profile 50 episodes of training
python scripts/profiler.py --target "python -m lunar-lander.main --episodes 50"

# Memory profile
python scripts/memory_profiler.py --target "python -m lunar-lander.main --episodes 20"

# Profile specific module
python scripts/profiler.py --target "python -c 'from training.trainer import TD3Trainer; ...'"
```

## Key Areas to Profile in RL

- `trainer.train_on_buffer()` - Training loop
- `replay_buffer.sample()` - Experience sampling
- `behavior_analyzer.analyze()` - Post-episode analysis
- `shape_reward()` - Per-step reward computation
- Neural network forward/backward passes

## Output Format

```
Top 20 functions by cumulative time:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1000    2.345    0.002   45.678    0.046 trainer.py:89(train_step)
    50000    1.234    0.000   12.345    0.000 replay_buffer.py:39(sample)
    ...

Bottleneck Analysis:
1. trainer.train_step - 45% of total time
   Suggestion: Batch operations, reduce tensor transfers
2. replay_buffer.sample - 12% of total time
   Suggestion: Use more efficient sampling strategy
```

## Profiling Modes

| Mode | Use When | Output |
|------|----------|--------|
| CPU (cProfile) | Finding slow functions | Function call times |
| Line | Optimizing specific function | Line-by-line timing |
| Memory | Tracking memory usage | Allocation over time |
| Statistical | Long-running code | Sampled call stacks |
