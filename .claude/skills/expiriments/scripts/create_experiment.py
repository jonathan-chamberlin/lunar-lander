#!/usr/bin/env python3
"""Create a new experiment folder with standard structure.

This script creates a new experiment in the lunar-lander/experiments/ directory
with the following structure:
    EXP_XXX_<name>/
        charts/
        results/
        frames/
        video/
        config.json
        EXPERIMENT.md

Usage:
    python create_experiment.py <experiment_name> [--id EXP_XXX]

Example:
    python create_experiment.py learning_rate_comparison
    python create_experiment.py learning_rate_comparison --id EXP_005
"""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path


def get_project_root() -> Path:
    """Get the lunar-lander-file-folder root directory."""
    script_path = Path(__file__).resolve()
    # Navigate up from .claude/skills/expiriments/scripts/ to root
    return script_path.parents[4]


def get_experiments_dir() -> Path:
    """Get the experiments directory path."""
    return get_project_root() / "lunar-lander" / "experiments"


def get_next_experiment_id(experiments_dir: Path) -> str:
    """Determine the next available experiment ID (e.g., EXP_005)."""
    existing_ids = []

    if experiments_dir.exists():
        for folder in experiments_dir.iterdir():
            if folder.is_dir():
                match = re.match(r"EXP_(\d+)", folder.name)
                if match:
                    existing_ids.append(int(match.group(1)))

    next_num = max(existing_ids, default=0) + 1
    return f"EXP_{next_num:03d}"


def create_experiment_md(exp_id: str, exp_name: str, title: str) -> str:
    """Generate the EXPERIMENT.md content from template."""
    today = date.today().isoformat()

    return f"""---
id: {exp_id}
title: {title}
status: PLANNED
created: {today}
completed:
concluded:
---

# Experiment: {title}

## Hypothesis

[State your hypothesis here - what do you expect to happen and why?]

**Reasoning:**
-
-
-

## Variables

### Independent (what you're changing)
- parameter_name: [value1, value2, value3]

### Dependent (what you're measuring)

**Standard metrics (ALWAYS report these):**
- Success rate (%)
- Max consecutive successes
- First success episode
- Final 100 success rate (%)
- Run time (seconds)
- Total successes
- Average env reward
- Final 100 average env reward

### Controlled (what stays fixed)
- All other hyperparameters (learning rate, buffer size, tau, etc.)
- Network architecture ([256, 128])
- Number of episodes per run (500)
- Noise decay schedule (300 episodes)

## Predictions

Make specific, falsifiable predictions before running:

- [ ] Prediction 1:
- [ ] Prediction 2:
- [ ] Prediction 3:

## Configuration

```json
{{
  "name": "{exp_name}",
  "type": "grid",
  "episodes_per_run": 500,
  "num_runs_per_config": 2,
  "parameters": {{
  }}
}}
```

## Execution

- **Started:**
- **Completed:**
- **Results folder:** `results/`
- **Charts folder:** `charts/`
- **Episodes per run:** 500
- **Number of configurations:**
- **Runs per configuration:** 2
- **Total runs:**

## Results

<!-- Fill after experiment completes -->

### Summary Table

| Run | Config | Success % | Max Consec | First Success | Final 100 % | Time | Total Successes | Avg Reward | Final 100 Reward |
|-----|--------|-----------|------------|---------------|-------------|------|-----------------|------------|------------------|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |

### Aggregated by Config

| Config | Avg Success % | Avg Final 100 % | Avg Time | Avg First Success |
|--------|---------------|-----------------|----------|-------------------|
| | | | | |

### Best Configuration

```json
{{
}}
```

### Key Observations

1.
2.
3.

### Charts

Charts for each run are saved in the `charts/` folder.

## Analysis

[What patterns emerged? Any surprises? How do results compare to predictions?]

### Prediction Outcomes

- [ ] Prediction 1: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 2: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]
- [ ] Prediction 3: **SUPPORTED / REFUTED / INCONCLUSIVE** - [explanation]

## Conclusion

### Hypothesis Status: **SUPPORTED / REFUTED / INCONCLUSIVE**

### Key Finding
[One sentence summarizing the main takeaway]

### Implications
[What does this mean for future training?]

### Next Steps
- [ ]
- [ ]
"""


def create_experiment(name: str, exp_id: str = None) -> Path:
    """Create a new experiment folder with standard structure.

    Args:
        name: Experiment name (e.g., "learning_rate_comparison")
        exp_id: Optional experiment ID (e.g., "EXP_005"). Auto-generated if not provided.

    Returns:
        Path to the created experiment folder.
    """
    experiments_dir = get_experiments_dir()

    # Ensure experiments directory exists
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Determine experiment ID
    if exp_id is None:
        exp_id = get_next_experiment_id(experiments_dir)

    # Sanitize name for folder (replace spaces with underscores, lowercase)
    folder_name = name.lower().replace(" ", "_").replace("-", "_")
    folder_name = re.sub(r"[^a-z0-9_]", "", folder_name)

    # Create experiment folder name
    exp_folder_name = f"{exp_id}_{folder_name}"
    exp_path = experiments_dir / exp_folder_name

    # Check if folder already exists
    if exp_path.exists():
        print(f"Error: Experiment folder already exists: {exp_path}", file=sys.stderr)
        sys.exit(1)

    # Create folder structure
    exp_path.mkdir(parents=True)
    (exp_path / "charts").mkdir()
    (exp_path / "results").mkdir()
    (exp_path / "frames").mkdir()
    (exp_path / "video").mkdir()

    # Create empty config.json
    config_path = exp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")

    # Create EXPERIMENT.md from template
    title = name.replace("_", " ").title()
    experiment_md = create_experiment_md(exp_id, folder_name, title)
    (exp_path / "EXPERIMENT.md").write_text(experiment_md, encoding="utf-8")

    return exp_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a new experiment folder with standard structure.",
        epilog="Example: python create_experiment.py learning_rate_comparison"
    )
    parser.add_argument(
        "name",
        help="Experiment name (e.g., 'learning_rate_comparison')"
    )
    parser.add_argument(
        "--id",
        dest="exp_id",
        help="Experiment ID (e.g., 'EXP_005'). Auto-generated if not provided."
    )

    args = parser.parse_args()

    exp_path = create_experiment(args.name, args.exp_id)

    print(f"Created experiment: {exp_path}")
    print(f"  - {exp_path / 'charts'}/")
    print(f"  - {exp_path / 'results'}/")
    print(f"  - {exp_path / 'frames'}/")
    print(f"  - {exp_path / 'video'}/")
    print(f"  - {exp_path / 'config.json'}")
    print(f"  - {exp_path / 'EXPERIMENT.md'}")


if __name__ == "__main__":
    main()
