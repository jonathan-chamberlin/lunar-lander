# 🚀 Lunar Lander TD3 - Deep Reinforcement Learning Agent

A reinforcement learning agent trained to land a spacecraft using **TD3 (Twin Delayed DDPG)**, optimized through 16 structured experiments from 0% success to **101 consecutive successful landings**.

The top left chart show how the agent's reward increased, and the bottom left chart shows the agent's consecutive successes. The chart shows that my algorithm beat the environement goal of 100 consecutive successes*

![Training Progress](https://github.com/jonathan-chamberlin/lunar-lander/blob/main/training_progress.png?raw=true)

https://github.com/user-attachments/assets/72bd66b4-2d8b-4d96-8e24-a9dc84cdc699


A success is when the agent scores 200 points or more.

---

## 📋 Overview

This project trains an agent to solve Gymnasium's **LunarLanderContinuous-v3** environment, a continuous control problem where the agent must fire a main thruster and side thrusters to safely land a spacecraft on a pad. The agent observes 8 values (position, velocity, angle, leg contact) and outputs 2 continuous actions (thrust commands) each timestep.

The algorithm is **TD3**, an actor-critic method designed for continuous action spaces. TD3 improves on earlier approaches (DDPG) by using twin critics to reduce value overestimation, delayed policy updates for stability, and target policy smoothing to prevent the critic from exploiting narrow peaks. The implementation uses PyTorch, with Prioritized Experience Replay and Ornstein-Uhlenbeck exploration noise.

---

## 🔬 The Experimental Approach

Rather than tuning hyperparameters by intuition, this project treats every design decision as a testable hypothesis. Each of the **16 experiments** follows the same structure: one independent variable is changed, all others are held constant, and results are measured against **8 standardized metrics** (success rate, max consecutive successes, first success episode, final-100 success rate, run time, total successes, average reward, and final-100 average reward). Every experiment has a written hypothesis, predictions, controlled configuration, and a post-run analysis.

This systematic approach uncovered several counterintuitive results that would have been missed by manual tuning:

- **Reward shaping can backfire.** Adding a small time penalty (-0.05 per step) to discourage hovering seemed reasonable, but it actually *tripled* the failure rate. Ablation testing revealed that the penalty created conflicting gradients. The agent learned to crash quickly to minimize accumulated penalty instead of learning to land. Removing the time penalty improved the final success rate from 16.5% to 51%.

- **Batch size has a unpredictable effect.** The default batch size of 128 caused complete training failure (0% success), while both smaller (32) and larger (256) batch sizes worked well. This suggests a specific interaction between batch size, update frequency, and the replay buffer's priority distribution.

- **Larger replay buffers can be catastrophic.** Conventional wisdom says bigger buffers give more diverse experience. But increasing the buffer from 16,384 to 65,536 caused total failure. The root cause: Prioritized Experience Replay (PER) kept sampling high-priority experiences from the agent's early, terrible policy, preventing it from learning from newer, better data. Smaller buffers naturally forget stale data.

The lesson repeated across experiments: **systematic testing reveals what intuition misses.**

---

## 🧪 Key Experiments

| # | Experiment | What Was Tested | Key Finding |
|---|---|---|---|
| EXP_012 | Reward Shaping Ablation | Which reward shaping components help vs. hurt | Time penalty *hurts* learning; removing it tripled success rate |
| EXP_007 | Buffer Size | Optimal replay buffer capacity | 16,384 optimal; larger buffers cause catastrophic staleness in PER |
| EXP_009 | Learning Rate Sweep | Actor vs. critic learning rate ratio | Equal rates (1:1) optimal at 68% final success; asymmetric ratios catastrophic |
| EXP_006 | Network Architecture | Small vs. large neural networks | [64, 32] outperforms [256, 128] (25.5% vs. 11.5%) while training 2.1x faster |
| EXP_004 | Batch Size | Optimal mini-batch size for training updates | Unpredictable: default 128 fails completely, while 32 and 256 both succeed |
| EXP_022 | Exploitation Phases | Decaying exploration noise to zero | Achieved **101 consecutive successes** with zero exploration noise |

All 16 experiments are documented in full in [`experiments/`](experiments/), each with hypothesis, configuration, raw results, and analysis.

---

## 📊 Results

Over 6,536 training episodes (~7 hours), the agent progressed through three distinct phases:

1. **Exploration (episodes 1-1,000):** The agent explores randomly with high noise. Success rate is near 0%, which is expected and necessary.
2. **Refinement (episodes 1,001-5,000):** As noise decays and the policy improves, success rate climbs to 57%, then 91% as the agent locks in on a reliable landing strategy.
3. **Exploitation (episodes 5,001+):** With exploration noise decayed to zero, the trained policy executes consistently. At episode 5,336, the agent achieved **101 consecutive successful landings**, well beyond the environment's standard benchmark of 200+ reward.

Peak success rate reached **91%** over the final training window. The training progress chart above shows the full learning curve, and the video above shows the agent's behavior evolving from erratic early attempts to clean, controlled landings.

---

## 🐛 Debugging and Lessons Learned

Two failures deepened my understanding of RL training dynamics more than any success:

- **The vectorized environment bug.** I assumed that training with 8 parallel environments would produce the same results as a single environment, just faster. It didn't. The parallel path produced 0% success while the single-environment path worked at 48%. Isolating the root cause (a subtle state-handling issue in the vectorized experience collection) was a lesson in never assuming two code paths behave the same just because they should.

- **The anti-hovering incident.** After making four simultaneous changes to reward shaping, the agent's success rate dropped to 0%. To understand why, I built a behavior categorization system that classifies 50+ flight patterns per episode. The categorization revealed that the agent was crashing on nearly every attempt, which led me to investigate the shaped reward. The agent had learned that crashing fast minimized the new penalties better than trying to land. This incident drove two outcomes: a personal rule of **one change at a time, with a maximum 2x magnitude adjustment**, and a diagnostic tool that became central to every experiment after.

Both incidents are documented in detail in [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md).

---

## 🏗️ Architecture and Engineering

- **TD3 with Prioritized Experience Replay** - priority sampling uses a SumTree data structure with JIT-compiled (Numba) batch operations for efficient O(log n) sampling
- **Ornstein-Uhlenbeck exploration noise** with configurable decay schedule from full exploration to zero noise
- **50+ behavior pattern detection** - automated classification of flight patterns (controlled descent, yo-yo, suicide burn, etc.) for diagnosing agent behavior
- **Hyperparameter sweep infrastructure** - grid search and random search with automated result aggregation
- **Custom reward shaping** gated on descent (prevents reward hacking through hovering)
- **Test suite, linting (Ruff), formatting (Black), and type checking (MyPy)**

---

## 📁 Project Structure

```
lunar-lander/
├── src/                    # Source code
│   ├── training/           #   TD3 trainer, networks, replay buffer, environment
│   ├── analysis/           #   Behavior analysis, diagnostics, charts
│   └── data/               #   Logging, aggregation, I/O
├── experiments/            # 16 structured experiments with full results
├── tools/                  # Sweep runner, profiler, log analyzer
├── tests/                  # Pytest test suite
├── docs/                   # Documentation (lessons learned, behaviors, codebase docs)
├── sweep_configs/          # Hyperparameter sweep configurations
└── simulations/            # Training run data and aggregates
```

---

## ▶️ How to Run

```bash
# Create and activate the virtual environment
python -m venv .venv-3.12.5
.venv-3.12.5/Scripts/activate     # Windows
# .venv-3.12.5/bin/activate       # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Run training
.venv-3.12.5/Scripts/python.exe src/main.py

# Run tests
pytest tests/
```

For detailed configuration options, sweep setup, and architecture documentation, see [`docs/`](docs/).
