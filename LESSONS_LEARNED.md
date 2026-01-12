# Lessons Learned: Reward Shaping Mistakes

## Incident: Anti-Hovering Changes Broke Training (2024)

### What Happened
The agent was exploiting reward shaping by hovering near the ground to accumulate bonuses. I implemented three "fixes" simultaneously that completely broke training:

1. Increased base time penalty from -0.05 to -0.3 (6x increase)
2. Added progressive time penalty: `penalty * (1 + step/200)`
3. Made descent requirement stricter: y_vel < -0.2 instead of -0.05

### The Result
- 0% success rate (was 7.6%)
- Agent learned to do NOTHING (99.9% no thrust)
- Just falls and crashes immediately
- Mean reward dropped from -174 to -595

### Why It Failed

**1. Penalties Too Severe**
- At step 200, penalty was -0.6/step
- At step 400, penalty was -0.9/step
- Agent couldn't distinguish "bad" from "less bad" - everything was terrible
- Optimal policy became: crash quickly to minimize accumulated penalty

**2. Made Multiple Changes Simultaneously**
- Each change alone might have worked
- Together they compounded into an unlearnable environment
- No way to know which change caused the problem

**3. Didn't Consider Learning Dynamics**
- Severe penalties don't teach good behavior
- They teach the agent to give up
- The agent needs a gradient to follow, not a cliff

**4. Broke the Reward Signal**
- Bonuses (max +5.8/step) couldn't overcome penalties (-0.3 to -1.8/step)
- Net reward was ALWAYS negative
- No positive signal to learn from

---

## Rules for Future Reward Shaping Changes

### Rule 1: Make ONE Change at a Time
- Test each modification independently
- Run at least 500 episodes before adding another change
- Document the effect of each change

### Rule 2: Preserve Positive Reward Potential
- Agent must be able to achieve positive rewards through good behavior
- If `max_possible_bonus < penalty`, the math is broken
- Calculate: can an optimal policy get positive reward?

### Rule 3: Incremental Magnitude Changes
- Never increase/decrease by more than 2x in one change
- Bad: -0.05 → -0.3 (6x increase)
- Good: -0.05 → -0.1 → -0.15 (gradual)

### Rule 4: Consider Episode Length
- A 200-step episode at -0.3/step = -60 penalty
- A 200-step episode at -0.05/step = -10 penalty
- Penalties accumulate! Small per-step = big total

### Rule 5: Test With Known-Good Parameters First
- Before blaming reward shaping, verify the base system works
- Keep a "known working" configuration to revert to

### Rule 6: Watch for "Give Up" Behavior
Signs the agent has given up:
- Thrust values near -1.0 or 1.0 constantly (saturated)
- Same action regardless of state
- Very short episodes (just falls)
- Success rate drops to 0%

---

## The Correct Approach for the Hovering Problem

Instead of severe penalties, should have tried (in order):

1. **First**: Just increase time penalty slightly: -0.05 → -0.1
2. **If still hovering**: Add a SMALL progressive component: `1 + step/500`
3. **If still hovering**: Make descent requirement slightly stricter: -0.05 → -0.1
4. **Nuclear option**: Remove per-step bonuses entirely, keep only terminal bonus

The key insight: hovering was profitable because bonuses > penalties.
Fix: make bonuses ≈ penalties, not penalties >>> everything.

---

## Recovery Action

Revert to the last known working configuration:
- Time penalty: -0.05 per step
- Descent requirement: y_vel < -0.05
- No progressive penalty

Then make SMALL, INCREMENTAL changes.
