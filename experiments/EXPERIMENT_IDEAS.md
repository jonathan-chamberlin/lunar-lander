# Experiment Ideas

Ideas for future experiments, organized by category. Parameter values chosen based on first-principles reasoning against current defaults.

**Current Defaults (from config.py):**
- batch_size: 128
- buffer_size: 16384
- actor_lr: 0.001, critic_lr: 0.002
- tau: 0.005
- hidden layers: [256, 128]
- noise_decay_episodes: 300
- max_episode_steps: 1000

---

## Performance Optimization Experiments

### EXP_004: Batch Size vs Training Speed

**Hypothesis:** Smaller batch sizes may improve learning in RL by providing more frequent, noisier gradient updates that help escape local minima.

**Reasoning:**
- Unlike supervised learning, larger batches aren't always better in RL
- Smaller batches = more frequent updates = faster adaptation to new experiences
- Current default (128) is mid-range; test both smaller and larger
- 512+ is likely overkill for this simple 8-state environment

```json
{
  "name": "batch_size_performance",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "batch_size": [32, 64, 128, 256]
  }
}
```

**Measure:** Episodes/second, success rate, learning stability

---

### EXP_005: Update Frequency

**Hypothesis:** Updating every 2-4 steps instead of every step will improve wall-clock speed with minimal impact on sample efficiency.

**Reasoning:**
- update_every=1 means gradient computation after every env step (expensive)
- Collecting multiple transitions before updating reduces overhead
- Too infrequent (16+) wastes collected data without learning
- Sweet spot likely 2-4 based on literature

```json
{
  "name": "update_frequency",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "update_every_n_steps": [1, 2, 4, 8]
  }
}
```

**Measure:** Wall-clock time per episode, success rate, time to first success

---

### EXP_006: Network Size vs Speed

**Hypothesis:** LunarLander's simple state space (8 dims) can be solved with much smaller networks than the current [256, 128].

**Reasoning:**
- Current network has ~35K parameters (actor) - likely overkill
- LunarLander is a solved benchmark; minimal networks should suffice
- [32, 32] = ~1.3K params - tests lower bound of viability
- Smaller networks = faster forward/backward passes

```json
{
  "name": "network_size_performance",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "hidden_sizes": [[32, 32], [64, 64], [128, 128], [256, 128]]
  }
}
```

**Measure:** Forward pass time (ms), training step time (ms), final success rate

---

### EXP_007: Replay Buffer Size

**Hypothesis:** Smaller replay buffers with more recent experiences may accelerate early learning by reducing distribution shift.

**Reasoning:**
- Current buffer (16384) holds ~160 episodes worth of data
- Smaller buffer = experiences more relevant to current policy
- Larger buffer = more diversity but older, possibly outdated experiences
- Tradeoff: stability vs adaptation speed

```json
{
  "name": "buffer_size",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "buffer_size": [8192, 16384, 32768, 65536]
  }
}
```

**Measure:** Memory usage, success rate, learning curve smoothness

---

### EXP_008: Episode Length Cap

**Hypothesis:** Capping episodes at 400-600 steps will save compute without hurting learning, since successful landings typically complete in 200-400 steps.

**Reasoning:**
- Default max is 1000 steps (gymnasium default)
- Skilled landings complete in 200-400 steps
- Failed episodes (floating, crashing slowly) waste compute going to 1000
- Cap too aggressive (<400) may terminate potentially successful attempts
- 400 is minimum viable; 600 provides safety margin

```json
{
  "name": "episode_length_cap",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "max_episode_steps": [400, 600, 800, 1000]
  }
}
```

**Measure:** Average steps per episode, episodes/second, success rate

---

## Learning Quality Experiments

### EXP_009: Learning Rate Sweep

**Hypothesis:** The optimal actor/critic learning rate ratio matters more than absolute values; critic should be 1-2x actor.

**Reasoning:**
- Current: actor=0.001, critic=0.002 (2x ratio)
- DDPG paper: actor=1e-4, critic=1e-3 (10x ratio)
- TD3 paper: both at 3e-4 (1x ratio)
- Critic learns Q-values (regression) - typically tolerates higher LR
- Actor learns policy (gradients through critic) - more sensitive
- Test values centered around current defaults

```json
{
  "name": "learning_rate_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 1,
  "parameters": {
    "actor_lr": [1e-4, 5e-4, 1e-3],
    "critic_lr": [5e-4, 1e-3, 2e-3]
  }
}
```

**Note:** 9 configs total; using 1 run per config to keep runtime manageable.

**Measure:** Convergence speed, final success rate, reward variance

---

### EXP_010: Noise Decay Schedule

**Hypothesis:** Extending exploration (slower noise decay) will improve final performance at cost of slower early convergence.

**Reasoning:**
- Current: noise decays over 300 episodes
- Fast decay (100 eps): Quick exploitation, risk of local minima
- Slow decay (500 eps): More exploration, slower convergence
- For 500-episode experiments, 300 means noise is minimal for final 200 episodes
- Testing if more sustained exploration helps

```json
{
  "name": "noise_decay_schedule",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "noise_decay_episodes": [100, 200, 300, 500]
  }
}
```

**Measure:** First success episode, exploration duration, final success rate

---

### EXP_011: Soft Update Rate (Tau)

**Hypothesis:** Higher tau values will accelerate learning but risk instability; optimal is likely between 0.005-0.01.

**Reasoning:**
- tau controls target network update: target = tau*online + (1-tau)*target
- DDPG paper: tau=0.001 (very conservative)
- Current default: tau=0.005 (moderate)
- Higher tau = faster target tracking = faster learning but potential oscillation
- 0.02+ likely causes instability in this environment

```json
{
  "name": "tau_sweep",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "tau": [0.001, 0.005, 0.01, 0.02]
  }
}
```

**Measure:** Learning stability (reward variance), convergence speed, final success rate

---

### EXP_012: Reward Shaping Ablation

**Hypothesis:** Some reward shaping components may be harmful to learning by creating local optima or conflicting gradients; a minimal shaping approach may outperform the full shaping function.

**Reasoning:**
- Current shaping has 4 per-step bonuses: time penalty, altitude bonus, leg contact bonus, stability bonus
- Each bonus adds learning signal but also risk of reward hacking or local optima
- Time penalty: Could rush agent into crashes, or helpfully discourage hovering
- Altitude bonus: Could cause diving behavior, or help guide descent
- Leg contact bonus: Could encourage premature landing attempts, or provide crucial signal
- Stability bonus: Could distract from position/velocity control, or improve final landings
- Testing all 2^4 = 16 combinations reveals which components actually help

**Current reward shaping components:**
```
1. time_penalty:    -0.05 per step (discourages hovering)
2. altitude_bonus:  +0.5 when y_pos < 0.25 AND descending (guides to ground)
3. leg_contact:     +2/+5 for one/both legs AND descending (rewards contact)
4. stability:       +0.3/+0.1 for |angle| < 0.1/0.2 AND descending (rewards upright)

(Terminal landing bonus +100 always enabled - this is the goal signal)
```

**Config parameters (booleans):**
- `reward_time_penalty`: Enable -0.05/step penalty
- `reward_altitude_bonus`: Enable +0.5 close-to-ground bonus
- `reward_leg_contact`: Enable +2/+5 leg contact bonuses
- `reward_stability`: Enable +0.3/+0.1 upright bonuses

```json
{
  "name": "reward_shaping_ablation",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {
    "reward_time_penalty": [false, true],
    "reward_altitude_bonus": [false, true],
    "reward_leg_contact": [false, true],
    "reward_stability": [false, true]
  }
}
```

**Note:** 2^4 = 16 configs × 2 runs = 32 total runs. This is a large experiment but essential for understanding which shaping signals help.

**Key configurations to watch:**
- `[false, false, false, false]`: No shaping (baseline - pure env reward)
- `[true, true, true, true]`: Full shaping (current default)
- `[true, false, false, false]`: Only time penalty
- `[false, false, true, false]`: Only leg contact (most direct landing signal)

**Measure:** Success rate, first success episode, final-100 success rate, learning stability

---

## Priority Queue

Experiments ordered by expected impact on performance:

1. **EXP_005: Update Frequency** - Highest potential speedup with minimal risk
2. **EXP_006: Network Size** - Could enable 2-4x faster forward passes
3. **EXP_008: Episode Length Cap** - Easy win for reducing wasted compute
4. **EXP_004: Batch Size** - May find faster config than current default
5. **EXP_007: Buffer Size** - May enable faster adaptation

Learning quality experiments (run after performance baseline established):

6. **EXP_009: Learning Rate Sweep** - Foundational tuning
7. **EXP_011: Tau Sweep** - Quick experiment, high potential impact
8. **EXP_010: Noise Decay** - Important for exploration/exploitation balance
