# Hyperparameter Ranges for TD3 Lunar Lander

## Recommended Ranges

### Learning Rates

| Parameter | Conservative | Moderate | Aggressive |
|-----------|--------------|----------|------------|
| `actor_lr` | 0.0001-0.0005 | 0.0005-0.002 | 0.002-0.01 |
| `critic_lr` | 0.0002-0.001 | 0.001-0.004 | 0.004-0.01 |

**Notes:**
- Critic LR typically 2x actor LR for stability
- Start conservative, increase if learning is slow
- Reduce if training is unstable (oscillating rewards)

### Batch Size

| Size | Memory | Training Speed | Stability |
|------|--------|----------------|-----------|
| 32 | Low | Fast updates | Less stable |
| 64 | Low | Fast updates | Moderate |
| 128 | Moderate | Balanced | Good |
| 256 | High | Slower updates | Very stable |
| 512 | High | Slowest | Most stable |

**Recommendation:** Start with 128, increase if unstable.

### Buffer Size

| Size | Memory | Sample Diversity | Correlation |
|------|--------|------------------|-------------|
| 8,192 | ~50MB | Lower | Higher |
| 16,384 | ~100MB | Moderate | Moderate |
| 32,768 | ~200MB | Good | Lower |
| 65,536 | ~400MB | High | Low |
| 131,072 | ~800MB | Very high | Very low |

**Notes:**
- Larger buffers reduce sample correlation
- Memory usage scales linearly
- For Lunar Lander, 16,384-32,768 is usually sufficient

### Discount Factor (gamma)

| Value | Horizon | Use Case |
|-------|---------|----------|
| 0.95 | ~20 steps | Short-term focus |
| 0.97 | ~33 steps | Moderate horizon |
| 0.99 | ~100 steps | Long-term planning |
| 0.995 | ~200 steps | Very long episodes |

**For Lunar Lander:** 0.99 works well (episodes ~200-1000 steps)

### Target Update Rate (tau)

| Value | Update Speed | Stability |
|-------|--------------|-----------|
| 0.001 | Very slow | Very stable |
| 0.005 | Slow | Stable |
| 0.01 | Moderate | Balanced |
| 0.02 | Fast | Less stable |

**Recommendation:** 0.005 is a safe default

### TD3-Specific: Policy Update Frequency

| Value | Policy Updates | Effect |
|-------|----------------|--------|
| 1 | Every critic update | Standard actor-critic |
| 2 | Every 2nd update | TD3 default |
| 3 | Every 3rd update | More conservative |
| 5 | Every 5th update | Very conservative |

**Notes:**
- Higher values = more stable but slower learning
- TD3 paper recommends 2
- For noisy environments, 3-5 can help

### Noise Parameters (OU Process)

| Parameter | Low | Default | High |
|-----------|-----|---------|------|
| `sigma` | 0.1 | 0.3 | 0.5 |
| `theta` | 0.1 | 0.2 | 0.3 |

**Noise Schedule:**
- `noise_scale_initial`: 1.0 (full exploration)
- `noise_scale_final`: 0.1-0.2 (minimal exploration)
- `noise_decay_episodes`: 200-500

## Quick Reference: Safe Starting Points

```python
# Conservative (stable but slow)
actor_lr = 0.0005
critic_lr = 0.001
batch_size = 256
buffer_size = 32768
gamma = 0.99
tau = 0.005
policy_update_frequency = 3

# Balanced (recommended)
actor_lr = 0.001
critic_lr = 0.002
batch_size = 128
buffer_size = 16384
gamma = 0.99
tau = 0.005
policy_update_frequency = 2

# Aggressive (fast but risky)
actor_lr = 0.002
critic_lr = 0.004
batch_size = 64
buffer_size = 16384
gamma = 0.99
tau = 0.01
policy_update_frequency = 2
```

## Sweep Priorities

When limited on compute, prioritize sweeping in this order:

1. **Learning rates** (highest impact)
2. **Batch size** (affects stability)
3. **Noise parameters** (affects exploration)
4. **Buffer size** (diminishing returns above 32k)
5. **Gamma/tau** (usually fine at defaults)
