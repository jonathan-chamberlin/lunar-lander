# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project for training an agent to control the LunarLanderContinuous-v3 environment from Gymnasium (formerly OpenAI Gym). The project uses PyTorch for building neural networks and Pygame for rendering the simulation.

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

Note: The project requires Box2D for the LunarLander physics simulation.

## Running the Simulation

The main simulation is in `main.py`:

```bash
python main.py
```

This runs multiple episodes of the lunar lander simulation with random actions. The simulation window can be closed by clicking the close button (event type 256), which terminates the current run.

## Project Planning

**plan.txt** - Task tracking file where pending tasks and completed tasks (when specified) are listed. Check this file to understand what the user is working on or planning to implement next.

## Code Architecture

### Main Components

**main.py** - Primary simulation file containing:
- Environment initialization using `gymnasium.make("LunarLanderContinuous-v3", render_mode="human")`
- Multi-run loop that executes `runs` number of episodes back-to-back
- Event handling for Pygame window (close event terminates simulation)
- Action sampling and environment stepping
- Reward tracking and accumulation per episode

**scratch.py** - Experimental/scratch file containing:
- `OUActionNoise` class (Ornstein-Uhlenbeck noise, commonly used for continuous action spaces in RL)
- Unrelated example code for learning Python classes

### Environment Details

The LunarLanderContinuous-v3 environment:
- **Observation space**: 8-dimensional vector containing position, velocity, angle, angular velocity, and leg contact information
- **Action space**: Continuous 2D action space (main engine thrust and side engine control)
- **Termination conditions**: Episode ends when `terminated=True` (lander crashes or lands)
- **Step return values**: `(observation, reward, terminated, truncated, info)`

### Current Implementation Flow

1. Initialize Gymnasium environment and Pygame
2. Get initial observation via `env.reset()[0]`
3. For each run:
   - Initialize empty reward list
   - Loop until episode terminates or window closes:
     - Process Pygame events (check for window close)
     - Sample random action from action space
     - Step environment with action
     - Collect reward and check termination status
   - Print accumulated reward for the run
4. When all runs complete or user closes window, exit

### Important Implementation Details

- **Reward tracking**: The `reward_list` is reset at the start of each run (main.py:41)
- **Event handling**: Pygame event type 256 is the window close event (main.py:47)
- **Episode reset**: When `terminated=True`, environment is reset but only the running loop exits (main.py:68-70)
- **Multi-run execution**: Simulation runs `runs` episodes consecutively until all complete or user closes window

## Development Notes

### Next Steps (Based on Commit History)

The codebase appears to be in early development stages. The commit history indicates:
- Basic environment setup is complete
- Random action sampling is working
- Reward tracking per episode is functional
- Next logical step would be implementing the Actor-Critic network (commented stub at main.py:33)

### Neural Network Architecture (To Be Implemented)

The commented line at main.py:33 suggests an Actor network will be added. For DDPG/TD3/SAC algorithms on continuous control:
- **Actor Network**: Maps observations (8D) to actions (2D continuous)
- **Critic Network**: Estimates Q-value for state-action pairs
- **Noise**: The `OUActionNoise` class in scratch.py is likely intended for exploration

### Key Files to Modify

When implementing the RL algorithm:
- Actor/Critic network classes should be added to main.py or a separate neural network module
- The action sampling line (main.py:51) should be replaced with actor network output
- Training loop with experience replay and gradient updates needs to be added
- The `OUActionNoise` class from scratch.py should be integrated for exploration

## Troubleshooting

### Virtual Environment Issues

The project has two venv directories (`.venv` and `.venv-3.12.5`). The active environment is `.venv-3.12.5` which contains all required packages. If imports fail, ensure you're using the correct Python interpreter.

### Pygame Warnings

You may see a deprecation warning about `pkg_resources` from Pygame. This is expected and can be ignored.
