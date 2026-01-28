# Statistical Tests for Run Comparison

## When You Need Statistics

Statistical tests are important when:
- Differences are small (<10% in success rate)
- You're making important decisions based on results
- You want to publish or share findings
- You're comparing across random seeds

## Basic Concepts

### Confidence Interval

**What it is:** Range likely to contain the true value

**For success rate (binomial proportion):**
```python
import math

def success_rate_ci(successes, total, confidence=0.95):
    """Wilson score interval for proportions."""
    if total == 0:
        return 0, 0

    z = 1.96  # 95% confidence
    p = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    spread = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator

    return max(0, center - spread), min(1, center + spread)

# Example: 78% success rate with 500 episodes
low, high = success_rate_ci(390, 500)
# Result: (0.742, 0.814) - true rate likely 74.2% to 81.4%
```

### Standard Error

**What it is:** Uncertainty in an estimate

```python
import math

def standard_error_mean(values):
    """Standard error of the mean."""
    n = len(values)
    std = statistics.stdev(values)
    return std / math.sqrt(n)

# Example: Mean reward 156.3 with SE 4.2
# True mean likely within 156.3 ± 8.4 (95% CI)
```

## Comparing Two Runs

### T-Test for Mean Rewards

**When:** Comparing if mean rewards are significantly different

```python
from scipy import stats

def compare_mean_rewards(rewards_a, rewards_b):
    """Compare mean rewards between two runs."""
    stat, p_value = stats.ttest_ind(rewards_a, rewards_b)

    return {
        'mean_a': sum(rewards_a) / len(rewards_a),
        'mean_b': sum(rewards_b) / len(rewards_b),
        't_statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Interpretation:
# p < 0.05: Statistically significant difference
# p >= 0.05: Difference may be due to chance
```

### Chi-Square Test for Success Rates

**When:** Comparing if success rates are significantly different

```python
from scipy import stats

def compare_success_rates(successes_a, total_a, successes_b, total_b):
    """Compare success rates between two runs."""
    # Create contingency table
    table = [
        [successes_a, total_a - successes_a],
        [successes_b, total_b - successes_b]
    ]

    chi2, p_value, dof, expected = stats.chi2_contingency(table)

    return {
        'rate_a': successes_a / total_a,
        'rate_b': successes_b / total_b,
        'chi2': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Example:
# Run A: 391/500 successes (78.2%)
# Run B: 412/500 successes (82.4%)
# Result: p = 0.11 - NOT significant (could be chance)
```

### Bootstrap Comparison

**When:** Comparing any metric without distributional assumptions

```python
import random

def bootstrap_compare(values_a, values_b, metric_func, n_bootstrap=10000):
    """Bootstrap comparison of any metric."""
    observed_diff = metric_func(values_a) - metric_func(values_b)

    # Combined pool
    combined = values_a + values_b
    n_a = len(values_a)

    # Generate bootstrap differences
    diffs = []
    for _ in range(n_bootstrap):
        random.shuffle(combined)
        sample_a = combined[:n_a]
        sample_b = combined[n_a:]
        diffs.append(metric_func(sample_a) - metric_func(sample_b))

    # P-value: proportion of bootstrap diffs >= observed
    p_value = sum(1 for d in diffs if abs(d) >= abs(observed_diff)) / n_bootstrap

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

## Comparing Multiple Runs

### ANOVA for Multiple Runs

**When:** Testing if any run is significantly different from others

```python
from scipy import stats

def compare_multiple_runs(runs_rewards):
    """Compare rewards across multiple runs."""
    stat, p_value = stats.f_oneway(*runs_rewards)

    return {
        'f_statistic': stat,
        'p_value': p_value,
        'any_significant': p_value < 0.05
    }

# If significant, use post-hoc tests to find which pairs differ
```

### Bonferroni Correction

**When:** Making multiple comparisons

```python
def adjusted_significance(n_comparisons, alpha=0.05):
    """Bonferroni-corrected significance level."""
    return alpha / n_comparisons

# Example: Comparing 4 runs (6 pairwise comparisons)
# Use p < 0.05/6 = 0.0083 for significance
```

## Practical Guidelines

### Sample Size Requirements

| Metric | Minimum Episodes | Recommended |
|--------|-----------------|-------------|
| Success rate | 100 | 500+ |
| Mean reward | 50 | 200+ |
| Variance estimates | 100 | 500+ |
| Learning curves | 200 | 500+ |

### Rule of Thumb

| Success Rate Difference | Episodes Needed | Likely Significant? |
|------------------------|-----------------|---------------------|
| >20% | 100 | Yes |
| 10-20% | 300 | Probably |
| 5-10% | 500+ | Maybe |
| <5% | 1000+ | Need stats |

### When Statistics Say "Not Significant"

This does NOT mean the runs are equal. It means:
- The difference MIGHT be real but we can't tell
- Need more data to be confident
- For practical purposes, treat them as similar

### When to Skip Statistics

- Differences are large (>20%)
- This is exploratory work
- You'll run more experiments anyway
- Resource constraints prevent proper sample sizes

## Quick Decision Guide

```
Is the difference > 20%?
├── Yes → Probably real, stats optional
└── No → Continue

Do you have 500+ episodes per run?
├── No → Collect more data or accept uncertainty
└── Yes → Continue

Run appropriate test:
├── Two runs, success rates → Chi-square test
├── Two runs, mean rewards → T-test
├── Multiple runs → ANOVA + post-hoc
└── Other metrics → Bootstrap

Is p < 0.05?
├── Yes → Difference is statistically significant
└── No → Difference may be due to chance
```

## Reporting Results

**Good:** "Run A achieved 82.4% success rate vs Run B's 78.2% (χ² = 2.54, p = 0.11), suggesting the difference may not be statistically significant."

**Bad:** "Run A is better with 82.4% vs 78.2% success rate."

Include:
- Actual values
- Sample sizes
- Test used
- P-value
- Whether significant
