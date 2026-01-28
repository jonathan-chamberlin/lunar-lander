---
name: expiriments
description: Creates and manages controlled experiments with fixed variables, tracked metadata, and repeatability. Use when user wants to create an experiment folder, setup an experiment, reference EXPERIMENT_IDEAS.md, or mentions EXP_ prefix.
---

# experiment

## Purpose

This skill exists **only** for running controlled experiments, not for normal simulation runs. Use it when the goal is to compare configurations, measure learning dynamics, or generate data for analysis and plots. Also use when asked to setup an expiriment.

Experiment runs answer: *"Why does it work (or fail), under which conditions, and how do changes affect outcomes?"*

Experiments require discipline: fixed variables, tracked metadata, repeatability, and clean outputs.

---

## When to Use This Skill

Use `experiment` **only if at least one of the following is true**:

1. You are asked to setup an expiriment.
2. You are comparing two or more configurations (hyperparameters, algorithms, noise schedules, env counts, etc.).
3. You are running sweeps, ablations, or controlled variations.
4. You need episode-level or run-level statistics saved for later analysis.
5. You care about reproducibility (same seed → same result).
6. You are producing charts, tables, or conclusions.

If none apply, **do not use this skill**. Use `simulation-execution` instead.

---

## Available Scripts

This skill includes helper scripts in the [scripts/](scripts/) folder:

### [create_experiment.py](scripts/create_experiment.py)

**ALWAYS use this script to create new experiments.** It automatically:
- Determines the next experiment ID
- Creates the folder structure (charts/, results/)
- Generates EXPERIMENT.md from template with proper formatting
- Creates empty config.json

```bash
# Usage (from project root - lunar-lander-file-folder/):
lunar-lander/.venv-3.12.5/Scripts/python.exe .claude/skills/expiriments/scripts/create_experiment.py <experiment_name> [--id EXP_XXX]

# Examples:
lunar-lander/.venv-3.12.5/Scripts/python.exe .claude/skills/expiriments/scripts/create_experiment.py update_frequency
lunar-lander/.venv-3.12.5/Scripts/python.exe .claude/skills/expiriments/scripts/create_experiment.py update_frequency --id EXP_005
```

---

## Experiment Workflow

All experiments follow this 5-phase workflow. **Do not skip phases.**

### Phase 1: PLAN

1. **Run the create_experiment.py script** to create the folder structure:
   ```bash
   lunar-lander/.venv-3.12.5/Scripts/python.exe .claude/skills/expiriments/scripts/create_experiment.py <experiment_name> [--id EXP_XXX]
   ```
   This automatically creates: folder, charts/, results/, config.json, and EXPERIMENT.md
2. Fill in EXPERIMENT.md with:
   - Hypothesis (specific, falsifiable)
   - Variables (independent, dependent, controlled)
   - Predictions (specific outcomes you expect)
3. Set status to `PLANNED`
4. Add entry to `experiments/INDEX.md`

### Phase 2: CONFIGURE

1. Create `config.json` in the experiment folder with the sweep configuration
2. Embed or reference the config in `EXPERIMENT.md`
3. Verify the config is valid with `--dry-run`:
   ```bash
   python tools/sweep_runner.py experiments/EXP_XXX_short_name/config.json --dry-run
   ```

### Phase 3: EXECUTE

1. Update status to `RUNNING` in `EXPERIMENT.md`
2. Run the sweep with output directed to the experiment folder:
   ```bash
   python tools/sweep_runner.py experiments/EXP_XXX_short_name/config.json --output-dir experiments/EXP_XXX_short_name/results
   ```
3. Update status to `COMPLETED` when finished
4. Record start/end times in `EXPERIMENT.md`

### Phase 4: ANALYZE

1. Read `results/summary.csv` and `results/all_results.json`
2. Fill in the Results section of `EXPERIMENT.md`:
   - Summary table with key metrics
   - Best configuration
   - Key observations
3. Evaluate each prediction: SUPPORTED / REFUTED / INCONCLUSIVE
4. Update status to `ANALYZED`

### Phase 5: CONCLUDE

1. Determine hypothesis status: SUPPORTED / REFUTED / INCONCLUSIVE
2. Write the key finding (one sentence)
3. Document implications and next steps
4. Update status to `CONCLUDED`
5. Update `experiments/INDEX.md` with the key finding

---

## Directory Structure

```
experiments/
  INDEX.md                              # Quick lookup of all experiments
  EXPERIMENT_TEMPLATE.md                # Template for new experiments
  EXP_001_lr_sweep/
    EXPERIMENT.md                       # Plan + Results + Conclusion
    config.json                         # Sweep configuration
    results/                            # All outputs from sweep_runner
      summary.csv
      all_results.json
      run_001_results.json
      ...
    charts/                             # Auto-generated charts for each run
      run_001_chart.png
      run_002_chart.png
      ...
  EXP_002_noise_ablation/
    EXPERIMENT.md
    config.json
    results/
    charts/
```

**Key principle:** One folder = one complete experiment. Everything stays together.

**Charts:** A comprehensive training progress chart is automatically generated for each run when the sweep completes. Charts include reward over time, success rate, outcome distribution, behavior analysis, and more.

---

## Running Sweeps

Use `sweep_runner.py` with the `--output-dir` flag to keep results in the experiment folder:

```bash
# Dry run (validate config)
python tools/sweep_runner.py experiments/EXP_001_lr_sweep/config.json --dry-run

# Execute (results go to experiment folder)
python tools/sweep_runner.py experiments/EXP_001_lr_sweep/config.json --output-dir experiments/EXP_001_lr_sweep/results
```

---

## Success Metrics (Dependent Variables)

These are the key metrics used to determine which configuration is optimal:

| Metric | Description |
|--------|-------------|
| Total successes | Count of episodes that achieved success |
| Success rate | Percentage of episodes that were successes |
| Max consecutive successes | Longest streak of successful episodes |
| Env reward | Raw environment reward (not shaped) |
| First success episode | Episode number of the first successful landing |
| Final 100 success rate | Success rate over the last 100 episodes |

---

## Core Principles

1. **Control** – Only one independent variable changes at a time.
   * Otherwise, causality is impossible to infer.

2. **Repeatability** – Every experiment must be reproducible from metadata alone.
   * Seeds, configs, and code version must be logged.

3. **Predictions First** – Write predictions BEFORE running.
   * Prevents retrofitting hypotheses to match results.

4. **Isolation** – Experiments must not mutate global state.
   * Each run is a fresh process.

5. **Comparability** – Outputs must be structured identically across runs.
   * Same metrics, same formats, same units.

---

## Required Metadata

Each experiment must record in `EXPERIMENT.md`:

* Experiment ID and title
* Hypothesis
* Independent and dependent variables
* Predictions (before running)
* Full configuration (embedded or linked)
* Start and end timestamps
* Results summary
* Conclusion with hypothesis status

---

## Anti-Patterns (Do NOT Do These)

* Skipping the PLAN phase and jumping straight to running
* Writing predictions after seeing results
* Changing multiple hyperparameters without justification
* Leaving experiments in COMPLETED status without analysis
* Not updating INDEX.md
* Running experiments outside the `experiments/` folder

---

## Interaction With Other Skills

* Uses **sweep** skill's `sweep_runner.py` to execute runs
* Does **not** handle rendering unless explicitly required
* Does **not** tune hyperparameters automatically unless instructed

---

## Summary

Use `experiment` when rigor matters.

Follow the 5-phase workflow: **PLAN → CONFIGURE → EXECUTE → ANALYZE → CONCLUDE**

Every experiment lives in its own folder with everything needed to understand and reproduce it.
