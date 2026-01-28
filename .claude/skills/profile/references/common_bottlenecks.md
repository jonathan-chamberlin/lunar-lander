# Common Bottlenecks in RL Code

## Training Loop Bottlenecks

### 1. Replay Buffer Sampling
**Symptom:** `replay_buffer.sample()` in top 5 by cumtime

**Causes:**
- O(n) operations on large buffers
- Copying data on every sample
- Creating new numpy arrays each call

**Solutions:**
```python
# Bad: Copying entire buffer
indices = np.random.choice(len(self.buffer), batch_size)
batch = [self.buffer[i] for i in indices]

# Good: Pre-allocated arrays, index directly
states = self.states[indices]
actions = self.actions[indices]
```

### 2. Neural Network Forward Pass
**Symptom:** `forward()` or `__call__` high in profile

**Causes:**
- Small batch sizes (overhead dominates)
- Unnecessary tensor operations
- CPU-GPU transfers

**Solutions:**
- Increase batch size
- Use `torch.no_grad()` for inference
- Keep tensors on same device
- Use `model.eval()` when not training

### 3. Gradient Computation
**Symptom:** `backward()` takes majority of time

**Causes:**
- Large networks
- Complex computation graphs
- Gradient accumulation

**Solutions:**
- Gradient checkpointing for memory
- Mixed precision training (fp16)
- Simpler network architectures

### 4. Target Network Updates
**Symptom:** Frequent spikes in per-episode time

**Causes:**
- Hard updates copying all parameters
- Too frequent updates

**Solutions:**
```python
# Bad: Hard update every step
if step % 100 == 0:
    target_net.load_state_dict(policy_net.state_dict())

# Good: Soft update with tau
for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

## Environment Bottlenecks

### 1. Environment Reset
**Symptom:** `env.reset()` shows high cumtime

**Causes:**
- Complex state initialization
- Loading assets/resources
- Random seed operations

**Solutions:**
- Pre-compute initial states
- Use vectorized environments
- Pool and reuse environments

### 2. Rendering
**Symptom:** `env.render()` dominates time

**Solutions:**
- Disable rendering during training
- Render only every N episodes
- Use headless rendering

### 3. Reward Computation
**Symptom:** Custom `shape_reward()` in hot path

**Solutions:**
- Simplify reward function
- Pre-compute static components
- Vectorize calculations

## Data Processing Bottlenecks

### 1. Observation Preprocessing
**Symptom:** `preprocess()` or normalization functions high

**Causes:**
- Creating new arrays
- Python loops over pixels
- Type conversions

**Solutions:**
```python
# Bad: Python loop
for i in range(len(obs)):
    obs[i] = obs[i] / 255.0

# Good: Vectorized
obs = obs.astype(np.float32) / 255.0
```

### 2. Action Processing
**Symptom:** `clip()` or `scale_action()` frequent

**Solutions:**
- Use in-place operations
- Pre-compute action bounds
- Batch action processing

## Memory-Related Bottlenecks

### 1. Garbage Collection
**Symptom:** Periodic slowdowns, "gc" in profile

**Causes:**
- Creating many temporary objects
- Large object allocations

**Solutions:**
- Pre-allocate buffers
- Reuse objects
- Disable GC during critical sections:
```python
import gc
gc.disable()
# ... training loop ...
gc.enable()
gc.collect()
```

### 2. Memory Fragmentation
**Symptom:** Gradual slowdown over time

**Solutions:**
- Use contiguous memory allocations
- Periodically defragment (reset buffers)

## I/O Bottlenecks

### 1. Logging
**Symptom:** `print()`, `logging`, or file writes in profile

**Solutions:**
- Reduce logging frequency
- Use buffered writing
- Log asynchronously

### 2. Checkpointing
**Symptom:** `torch.save()` or similar in profile

**Solutions:**
- Save less frequently
- Save asynchronously
- Compress checkpoints

## Quick Optimization Checklist

| Bottleneck | Quick Fix |
|------------|-----------|
| Buffer sampling | Pre-allocate, use numpy indexing |
| Forward pass | Increase batch size, use eval mode |
| Backward pass | Mixed precision, gradient checkpointing |
| Environment | Vectorize, disable rendering |
| Preprocessing | Vectorize, avoid copies |
| GC pauses | Pre-allocate, disable during training |
| Logging | Reduce frequency, buffer writes |

## Profiling Tips for RL

1. **Profile short runs first** - 50-100 episodes is enough
2. **Profile representative load** - Include all phases (exploration, learning)
3. **Check both CPU and memory** - Memory issues cause CPU stalls
4. **Profile on target hardware** - GPU bottlenecks differ from CPU
5. **Re-profile after changes** - Optimizing one thing may reveal new bottlenecks
