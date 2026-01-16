# Data Access Policy

This document defines the data tiers and access patterns for the lunar lander training data architecture.

## Per-Run Data: Memory vs Disk

This section clarifies what data exists in memory during simulation versus what gets written to disk files for later analysis.

### In-Memory Only (NOT written to disk)

These are held in RAM during an episode but discarded after the episode ends:

| Data | Size per Episode | Purpose |
|------|------------------|---------|
| Raw observations array | ~8 floats × 200-1000 steps | Behavior analysis, then discarded |
| Raw actions array | ~2 floats × 200-1000 steps | Behavior analysis, then discarded |
| Step-by-step rewards | ~200-1000 floats | Summed to env_reward, then discarded |
| Q-value predictions | Per training step | Used for training only |
| Gradients | Per training step | Used for training only |

**Why not written to disk**: Full trajectories would increase runs.jsonl from ~500 bytes/episode to ~10 KB/episode (20x larger). Summary metrics are sufficient for most analysis.

### Written to Disk (persisted in runs.jsonl)

Each episode appends one JSON line to `runs.jsonl` containing **aggregated summary metrics**, not raw step-by-step data. For example, instead of storing 500 individual reward values, we store the single summed `env_reward`. Instead of storing 500 observation vectors, we store the derived `outcome` and `behaviors` classifications.

Fields written per episode:

| Field | Type | Description |
|-------|------|-------------|
| `run_number` | int | Zero-indexed episode number |
| `env_reward` | float | Raw environment reward (no shaping) |
| `shaped_bonus` | float | Additional reward from shaping |
| `total_reward` | float | env_reward + shaped_bonus |
| `steps` | int | Number of steps in episode |
| `duration_seconds` | float | Wall-clock time for episode |
| `success` | bool | Whether reward >= success_threshold |
| `outcome` | string | Classification (e.g., 'LANDED_SAFE', 'CRASHED') |
| `behaviors` | list | Detected behaviors (e.g., ['STAYED_UPRIGHT', 'CONTROLLED_DESCENT']) |
| `terminated` | bool | Episode ended due to terminal state |
| `truncated` | bool | Episode was truncated (time limit) |
| `rendered` | bool | Whether this episode was rendered |
| `timestamp` | string | ISO 8601 timestamp when run completed |

**Note**: If trajectory replay or detailed debugging is needed in the future, optional trajectory file storage can be added.

## Data Tiers

### RAW (Immutable)
- **Location**: `simulations/{id}/runs.jsonl`
- **Content**: One JSON line per episode with summary metrics (see above)
- **Access**: Append-only during simulation, read-only afterwards
- **Use Case**: Complete audit trail, debugging, post-hoc analysis

### DERIVED (Computed)
- **Location**: `simulations/{id}/aggregates/*.json`
- **Content**: Computed statistics from raw data
- **Access**: Written periodically and at simulation end
- **Use Case**: Quick access to summary statistics without loading raw data

### ARTIFACTS (Regenerable)
- **Location**: `simulations/{id}/charts/`, `simulations/{id}/text/`
- **Content**: Visualizations and formatted reports
- **Access**: Can be regenerated from derived data
- **Use Case**: Human-readable outputs

## Agent Access Guidelines

### Recommended Access Pattern

Agents analyzing training results should access data in this priority order:

1. **Experiments Summary** (highest priority)
   - `experiments/{name}/summary.json`
   - Contains cross-simulation aggregate statistics
   - Sufficient for comparing training runs

2. **Simulation Aggregates**
   - `simulations/{id}/aggregates/final.json`
   - Contains complete statistics for one simulation
   - Sufficient for analyzing individual training runs

### Discouraged Access

Agents should NOT directly access:

1. **Raw Run Logs**
   - `simulations/{id}/runs.jsonl`
   - Too verbose for agent context windows
   - Use aggregates instead

2. **Chart Images**
   - `simulations/{id}/charts/*.png`
   - Require visual processing
   - Statistics are available in aggregates

3. **Periodic Batch Files**
   - `simulations/{id}/aggregates/batch_*.json`
   - Intermediate snapshots for recovery
   - Use final.json for analysis

## Data Flow

```
Episode Completion
       |
       v
+------------------+
| RunLogger        | --> runs.jsonl (RAW)
| (append-only)    |
+------------------+
       |
       v
+------------------+
| DiagnosticsTracker|
| (incremental)    |
+------------------+
       |
       v (every 100 episodes)
+------------------+
| AggregateWriter  | --> aggregates/batch_NNNN.json
+------------------+
       |
       v (simulation end)
+------------------+
| AggregateWriter  | --> aggregates/final.json (DERIVED)
+------------------+
       |
       v (experiment creation)
+------------------+
| ExperimentDir    | --> experiments/{name}/summary.json
+------------------+
```

## File Sizes (Approximate)

| File | Size per 1000 episodes |
|------|------------------------|
| runs.jsonl | ~500 KB |
| final.json | ~10 KB |
| summary.json | ~5 KB |

## Query Patterns

### "What was the success rate?"
- Access: `experiments/{name}/summary.json` or `simulations/{id}/aggregates/final.json`
- Field: `summary.success_rate`

### "Which simulation performed best?"
- Access: `experiments/{name}/summary.json`
- Field: `best_simulation.by_success_rate`

### "What behaviors correlated with success?"
- Access: `simulations/{id}/aggregates/final.json`
- Field: `behaviors.success_correlation`

### "What was the outcome distribution?"
- Access: `simulations/{id}/aggregates/final.json`
- Field: `behaviors.outcome_category_counts`

## Immutability Guarantees

1. **Config Snapshot**: Written once at simulation start, never modified
2. **Runs Log**: Append-only, no modifications or deletions
3. **Simulation ID**: Unique per simulation (timestamp + UUID)

## Recovery

If a simulation is interrupted:
1. Raw data in runs.jsonl is preserved
2. Last periodic aggregate shows progress
3. Final aggregate will be missing but can be regenerated from runs.jsonl

## Security Considerations

- No sensitive data in any tier
- Config snapshots may contain hyperparameters
- Model weights stored separately in `models/`
