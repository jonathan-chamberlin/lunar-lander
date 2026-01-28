---
name: speed-up
description: Find slow code patterns and suggest faster alternatives. Use when user mentions performance, slow code, optimization, or wants to make code faster. Identifies unnecessarily slow methods, loops that could be vectorized, and inefficient data structures.
allowed-tools: Read, Grep, Glob, Bash
---

# Speed-Up: Find and Fix Slow Code Patterns

Identify code patterns that are unnecessarily slow and suggest faster alternatives.

## When to Use This Skill

- User mentions code is "slow" or wants to "speed up" something
- User asks about "optimization" or "performance"
- User wants to find "bottlenecks" or "inefficiencies"
- Before major refactoring for performance

## Analysis Workflow

1. **Scan codebase** for known slow patterns using `scripts/slow_pattern_finder.py`
2. **Identify vectorization opportunities** for numpy conversions
3. **Suggest specific replacements** with expected speedup

## Reference Documentation

- For slow vs fast pattern catalog, see [slow_vs_fast.md](references/slow_vs_fast.md)
- For numpy vectorization patterns, see [numpy_vectorization.md](references/numpy_vectorization.md)

## Common Slow Patterns to Find

### Python Anti-Patterns
- `for` loops that could be list comprehensions
- Repeated `.append()` in loops (use list comprehension)
- String concatenation in loops (use `''.join()`)
- `in` checks on lists (use sets)
- Repeated dictionary key access (cache in variable)
- Cache misses

### NumPy Anti-Patterns
- Python loops over numpy arrays
- Element-wise operations instead of vectorized
- Repeated array creation inside loops
- Using `.tolist()` unnecessarily

### RL-Specific Anti-Patterns
- Copying entire replay buffer for sampling
- Recomputing static values every step
- Unnecessary tensor device transfers
- Synchronous operations that could be batched

## Scripts

- `scripts/slow_pattern_finder.py` - Scan for known slow patterns
- `scripts/benchmark.py` - Compare before/after performance
