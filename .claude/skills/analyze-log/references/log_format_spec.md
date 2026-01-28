# Training Log Format Specification

## Episode Result Line Format

The standard format for episode results:

```
Run {episode} {status} {outcome} {emoji} {message} 🥕 Reward: {reward} (env: {env_reward} / shaped: {shaped_reward})
```

### Components

| Component | Format | Example |
|-----------|--------|---------|
| `episode` | Integer | `123` |
| `status` | ✓ or ✗ | `✓` = success, `✗` = failure |
| `outcome` | SCREAMING_SNAKE_CASE | `LANDED_SOFTLY` |
| `emoji` | ✅ or ❌ | ✅ = success, ❌ = failure |
| `message` | Human-readable | `Landed Safely` |
| `reward` | Float | `245.3` |
| `env_reward` | Float | `250.1` |
| `shaped_reward` | Float (signed) | `-4.8` or `+10.8` |

### Examples

```
Run 1 ✗ CRASHED_HIGH_VELOCITY ❌ Didn't land safely 🥕 Reward: -89.2 (env: -100.0 / shaped: +10.8)
Run 2 ✓ LANDED_SOFTLY ✅ Landed Safely 🥕 Reward: 245.3 (env: 250.1 / shaped: -4.8)
Run 50 ✓ LANDED_PERFECTLY ✅ Landed Safely 🥕 Reward: 298.5 (env: 300.0 / shaped: -1.5)
```

## Outcome Types

### Success Outcomes (✓)

| Outcome | Description | Typical Reward |
|---------|-------------|----------------|
| `LANDED_PERFECTLY` | Ideal landing, minimal velocity | 280-300+ |
| `LANDED_SOFTLY` | Good landing, low velocity | 200-280 |
| `LANDED_HARD` | Successful but rough | 100-200 |

### Failure Outcomes (✗)

| Outcome | Description | Typical Reward |
|---------|-------------|----------------|
| `CRASHED_HIGH_VELOCITY` | Hit ground too fast vertically | -100 to -50 |
| `CRASHED_FAST_HORIZONTAL` | Hit ground with lateral speed | -100 to -50 |
| `CRASHED_TILTED` | Landed at bad angle | -100 to -50 |
| `FLEW_OFF_LEFT` | Exited bounds left | -100 to -200 |
| `FLEW_OFF_RIGHT` | Exited bounds right | -100 to -200 |
| `FLEW_OFF_TOP` | Exited bounds top | -100 to -200 |
| `TIMEOUT` | Exceeded step limit | -50 to 0 |

## Success Criteria

An episode is considered successful if:
- Reward >= 200 (standard threshold)
- Status indicator is ✓
- Outcome contains "LANDED"

## Reward Components

| Component | Range | Description |
|-----------|-------|-------------|
| Base reward | -200 to +300 | From environment |
| Shaped reward | -50 to +50 | Additional shaping |
| Total reward | -250 to +350 | Sum displayed |

### Reward Breakdown

```
Total Reward = Environment Reward + Shaped Reward

Environment Reward:
  +100 to +140: Leg contact bonus
  +100: Successful landing
  -100: Crash
  Per-step: Fuel usage, position penalty

Shaped Reward (varies by implementation):
  Velocity bonus/penalty
  Angle bonus/penalty
  Distance to pad bonus
```

## Additional Log Lines

### Training Progress

```
Episode 100/1000 | Buffer: 5000 | Epsilon: 0.85
```

### Periodic Summary

```
=== Episodes 100-150 Summary ===
Success Rate: 45.2%
Mean Reward: 125.3
Best: 287.5 | Worst: -156.2
```

### Configuration

```
Training Config:
  actor_lr: 0.001
  critic_lr: 0.002
  batch_size: 128
```

## Parsing Notes

1. **Encoding**: Files may contain UTF-8 emoji characters
2. **Line endings**: Handle both Unix (LF) and Windows (CRLF)
3. **Optional fields**: Some logs may omit `env`/`shaped` breakdown
4. **Noise**: Logs may contain debug output, warnings, etc.

## Minimal Pattern for Extraction

For basic parsing, this regex captures essential data:

```python
pattern = r"Run (\d+)\s*([✓✗])\s*(\w+).*Reward:\s*([\d.-]+)"
# Groups: episode, status, outcome, reward
```
