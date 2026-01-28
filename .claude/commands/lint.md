---
description: Format and lint the codebase
---

Run code quality tools on the lunar lander project.

## Tools

1. **black** - Auto-format Python code (PEP 8 compliant)
2. **ruff** - Fast linter (replaces flake8/pylint)
3. **mypy** - Static type checking

## Steps

1. Check if tools are installed (`pip list | grep -E "black|ruff|mypy"`)
2. Install missing tools if needed
3. Ensure `pyproject.toml` has tool configuration
4. Run each tool and report issues
5. Optionally auto-fix (black, ruff --fix)

## Commands

```bash
# Check formatting (no changes)
black --check lunar-lander/

# Auto-format
black lunar-lander/

# Lint with ruff
ruff check lunar-lander/

# Auto-fix lint issues
ruff check lunar-lander/ --fix

# Type checking
mypy lunar-lander/ --ignore-missing-imports
```

## pyproject.toml Configuration

If not present, add this configuration:

```toml
[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP", "B", "C4"]
ignore = ["E501"]  # Line too long (handled by black)

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
ignore_missing_imports = true
```

## Expected Issues

Common issues in RL codebases:
- Type annotations missing for numpy arrays/tensors
- Line length (configurable)
- Import ordering
- Unused variables (often intentional in RL)

## Auto-Fix Mode

When user requests auto-fix:
1. Run `black lunar-lander/` to format
2. Run `ruff check lunar-lander/ --fix` to fix lint issues
3. Report remaining issues that need manual attention
