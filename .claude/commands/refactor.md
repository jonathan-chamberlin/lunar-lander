---
description: Review code structure and identify refactoring opportunities
---

Review all files again to see where the code could be put closer to it's ideal size and or shape. Analyze the codebase for refactoring opportunities. Be specific and actionable.

## Metrics to Enforce
- Functions: ≤40 lines, ≤4 parameters, cyclomatic complexity ≤10
- Classes: ≤300 lines, single responsibility
- Modules: ≤500 lines, cohesive purpose

## Code Smells to Detect
- God classes / functions doing too much
- Feature envy (method uses another class's data excessively)
- Long parameter lists (→ parameter objects)
- Duplicate logic (→ extract shared utility)
- Primitive obsession (→ value objects)
- Shotgun surgery (one change = edits across many files)
- Deep nesting (≥3 levels → early returns or extraction)

## RL-Specific Concerns
- Agent/environment coupling (should be interface-driven)
- Replay buffer abstraction leaks
- Policy/value network separation
- Config hardcoding vs. injectable hyperparameters

## Output Format
For each opportunity:
```
### [Priority: HIGH/MED/LOW] <Summary>
- **File:** `path/to/file.py:L##-L##`
- **Smell:** <which code smell>
- **Refactor:** <specific pattern to apply>
- **Risk:** <breaking change? test impact?>
- **Effort:** <small/medium/large>
```

List opportunities first. Do not modify code until I approve.
