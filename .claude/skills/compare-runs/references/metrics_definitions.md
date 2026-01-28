# Metrics Definitions

## Primary Metrics

### Success Rate

**Definition:** Percentage of episodes where reward >= 200

**Formula:**
```
Success Rate = (Episodes with reward >= 200) / (Total episodes) × 100%
```

**Interpretation:**
- 80%+: Excellent - agent reliably succeeds
- 60-80%: Good - agent usually succeeds
- 40-60%: Moderate - agent sometimes succeeds
- <40%: Poor - agent needs more training

**Considerations:**
- Threshold of 200 is specific to Lunar Lander
- May vary for other environments

---

### Mean Reward

**Definition:** Average reward across all episodes

**Formula:**
```
Mean Reward = Σ(episode rewards) / N
```

**Interpretation:**
- >200: Agent succeeds on average
- 100-200: Agent partially succeeds
- 0-100: Agent performs basic tasks
- <0: Agent fails more than succeeds

**Considerations:**
- Can be skewed by outliers
- Use with standard deviation for full picture

---

### Max Reward

**Definition:** Highest single-episode reward achieved

**Interpretation:**
- Shows agent's peak capability
- High max with low mean suggests inconsistency
- Useful for identifying potential

---

### First Success Episode

**Definition:** Episode number of first reward >= 200

**Interpretation:**
- Lower is better (faster learning)
- <50: Very fast learning
- 50-100: Normal learning speed
- 100-200: Slow but learning
- >200 or N/A: May need hyperparameter tuning

**Considerations:**
- Single data point - can be lucky/unlucky
- More meaningful with multiple seeds

---

### Final N Success Rate

**Definition:** Success rate of last N episodes (typically N=100)

**Formula:**
```
Final N Success Rate = (Successes in last N episodes) / N × 100%
```

**Interpretation:**
- Shows convergent performance
- Higher than overall = still improving
- Lower than overall = performance degraded
- Use for comparing "final" capability

---

### Standard Deviation

**Definition:** Spread of rewards around the mean

**Formula:**
```
σ = √[Σ(reward - mean)² / (N-1)]
```

**Interpretation:**
- Low (< 50): Consistent performance
- Medium (50-100): Normal variance
- High (> 100): Inconsistent - may indicate instability

---

## Secondary Metrics

### Learning Speed

**Definition:** Episodes needed to reach sustained success (e.g., 50% success rate)

**Measurement:**
```python
def learning_speed(episodes, target_rate=0.5, window=50):
    for i in range(window, len(episodes)):
        window_episodes = episodes[i-window:i]
        if success_rate(window_episodes) >= target_rate:
            return i
    return -1  # Never reached
```

---

### Stability

**Definition:** Variance in success rate over time

**Measurement:**
```python
def stability(episodes, window=50):
    rates = [success_rate(episodes[i-window:i])
             for i in range(window, len(episodes))]
    return 1 - std(rates)  # Higher = more stable
```

---

### Sample Efficiency

**Definition:** Performance achieved per episode of training

**Formula:**
```
Sample Efficiency = Success Rate / Episodes Trained
```

**Use:** Comparing algorithms or hyperparameters with different training lengths

---

## Metric Relationships

| If... | And... | Then... |
|-------|--------|---------|
| High success rate | Low std | Stable, reliable agent |
| High success rate | High std | Lucky runs, needs more training |
| Low success rate | Fast first success | Early success but didn't sustain |
| High mean reward | Low success rate | Close misses (rewards 150-199) |
| High final N rate | Lower overall | Agent improved during training |
| Low final N rate | Higher overall | Performance degraded |

## Comparing Runs

### Fair Comparison Requirements

1. **Same episode count** - Or normalize metrics
2. **Same environment** - Including seeds if deterministic
3. **Same success threshold** - Consistent definition
4. **Same evaluation method** - Training vs evaluation mode

### Ranking Priorities

1. **Final success rate** - Most important for deployment
2. **Mean reward** - Overall performance
3. **Stability** - Consistency matters
4. **Learning speed** - For iteration speed
5. **Sample efficiency** - For resource-constrained settings

### When Differences Matter

| Metric | Meaningful Difference |
|--------|----------------------|
| Success Rate | > 5% (with sufficient episodes) |
| Mean Reward | > 20 points |
| First Success | > 20 episodes |
| Std Deviation | > 20 points |

For close comparisons, run multiple seeds and use statistical tests.
