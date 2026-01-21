# Implementation Plan: Modular Reward Shaping

## Objective
Refactor reward shaping to be modular and configurable, allowing easy testing of different reward component combinations.

## Design Principles
1. **Minimal changes** - Only modify what's necessary
2. **Modular structure** - Easy to add/remove reward behaviors
3. **Config-driven** - All toggles via config, testable via sweep runner

---

## Files to Modify

### 1. `src/config.py` - Add RewardShapingConfig

**Change:** Add new dataclass with boolean flags for each reward component.

```python
@dataclass(frozen=True)
class RewardShapingConfig:
    """Configuration for reward shaping components.

    Each boolean enables/disables a specific reward shaping behavior.
    Terminal landing bonus (+100) is always enabled as the goal signal.
    """

    time_penalty: bool = True      # -0.05/step (discourages hovering)
    altitude_bonus: bool = True    # +0.5 when low and descending
    leg_contact: bool = True       # +2/+5 for one/both legs when descending
    stability: bool = True         # +0.3/+0.1 for upright when descending
```

**Also:** Add `reward_shaping: RewardShapingConfig` to master `Config` class.

**Lines affected:** ~10 new lines, 1 line modified

---

### 2. `src/training/environment.py` - Refactor shape_reward()

**Change:** Accept config parameter, conditionally apply each component.

**Current signature:**
```python
def shape_reward(state, base_reward, terminated, step=0) -> float:
```

**New signature:**
```python
def shape_reward(state, base_reward, terminated, config: RewardShapingConfig, step=0) -> float:
```

**New structure (modular):**
```python
def shape_reward(
    state: np.ndarray,
    base_reward: float,
    terminated: bool,
    config: RewardShapingConfig,
    step: int = 0
) -> float:
    """Apply configurable reward shaping."""
    shaped_reward = base_reward

    y_pos, y_vel, angle = state[1], state[3], state[4]
    leg1_contact, leg2_contact = state[6], state[7]
    is_descending = y_vel < -0.05

    # Component 1: Time penalty
    if config.time_penalty:
        shaped_reward -= 0.05

    # Components 2-4 only apply when descending
    if is_descending:
        # Component 2: Altitude bonus
        if config.altitude_bonus and y_pos < 0.25:
            shaped_reward += 0.5

        # Component 3: Leg contact bonus
        if config.leg_contact:
            if leg1_contact and leg2_contact:
                shaped_reward += 5
            elif leg1_contact or leg2_contact:
                shaped_reward += 2

        # Component 4: Stability bonus
        if config.stability:
            if abs(angle) < 0.1:
                shaped_reward += 0.3
            elif abs(angle) < 0.2:
                shaped_reward += 0.1

    # Terminal bonus (ALWAYS enabled - this is the goal)
    if terminated and leg1_contact and leg2_contact and base_reward > 0:
        shaped_reward += 100

    return shaped_reward
```

**Lines affected:** ~15 lines modified (same logic, just conditionals added)

---

### 3. `src/training/runner.py` - Update call site

**Change:** Pass reward shaping config to shape_reward().

**Current (line 315):**
```python
shaped_reward = shape_reward(obs, reward, terminated, step=current_step)
```

**New:**
```python
shaped_reward = shape_reward(obs, reward, terminated, config.reward_shaping, step=current_step)
```

**Lines affected:** 1 line

---

### 4. `src/training/__init__.py` - No changes needed

The import already exports `shape_reward`, signature change is compatible.

---

## Adding New Reward Behaviors (Future)

To add a new reward component (e.g., "velocity_penalty"):

1. **config.py**: Add `velocity_penalty: bool = True` to `RewardShapingConfig`
2. **environment.py**: Add conditional block in `shape_reward()`:
   ```python
   if config.velocity_penalty:
       shaped_reward -= abs(x_vel) * 0.1
   ```

That's it - 2 lines of code to add a new toggleable behavior.

---

### 5. `tools/sweep_runner.py` - Add reward_shaping support

**Change:** Update `apply_params_to_config()` to handle RewardShapingConfig fields.

**Add after line 153:**
```python
reward_shaping_fields = set(RewardShapingConfig.__dataclass_fields__.keys())
```

**Add in the for loop (after line 161):**
```python
elif name in reward_shaping_fields:
    reward_shaping_params[name] = value
```

**Add after line 168:**
```python
new_reward_shaping = replace(config.reward_shaping, **reward_shaping_params) if reward_shaping_params else config.reward_shaping
```

**Update return (line 170-176):**
```python
return Config(
    training=new_training,
    noise=new_noise,
    run=new_run,
    environment=config.environment,
    display=config.display,
    reward_shaping=new_reward_shaping
)
```

**Lines affected:** ~8 lines

---

## Sweep Config Format

Use flat parameter names (no nesting):

```json
{
  "parameters": {
    "time_penalty": [false, true],
    "altitude_bonus": [false, true],
    "leg_contact": [false, true],
    "stability": [false, true]
  }
}
```

These map directly to `RewardShapingConfig` fields.

---

## Summary of Changes

| File | Lines Changed | Type |
|------|---------------|------|
| `config.py` | +12 | New dataclass + 1 line in Config |
| `environment.py` | ~15 | Add conditionals to existing logic |
| `runner.py` | 1 | Pass config to function call |
| **Total** | **~28 lines** | Minimal refactor |

---

## Testing

After implementation:
1. Run with all defaults (should behave identically to current)
2. Run with all disabled (pure env reward)
3. Run EXP_012 sweep to test all 16 combinations
