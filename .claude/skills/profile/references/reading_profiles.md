# Reading and Interpreting Profile Output

## cProfile Output Columns

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.050    0.001    2.500    0.025 trainer.py:89(train_step)
```

| Column | Meaning |
|--------|---------|
| `ncalls` | Number of times function was called |
| `tottime` | Total time spent IN this function (excluding subfunctions) |
| `percall` | `tottime / ncalls` |
| `cumtime` | Cumulative time (including all subfunctions) |
| `percall` | `cumtime / ncalls` |
| `filename:lineno(function)` | Where the function is defined |

## Key Metrics to Focus On

### 1. Cumulative Time (cumtime)
**Most important for finding bottlenecks.**

High cumtime means this function (including everything it calls) takes a lot of time.

```
# This is your bottleneck
    1000    0.100    0.000   45.000    0.045 main_loop()
```

### 2. Total Time (tottime)
**Important for finding functions doing heavy work themselves.**

High tottime with low cumtime = the function itself is slow (not its children).

```
# This function itself is slow
   50000   12.000    0.000   12.500    0.000 heavy_computation()
```

### 3. Call Count (ncalls)
**High call counts may indicate optimization opportunities.**

```
# Called way too many times
10000000    5.000    0.000    5.000    0.000 tiny_helper()
```

Consider:
- Caching results
- Batching calls
- Inlining the function

## Reading Patterns

### Pattern 1: Hot Function
```
   ncalls  tottime  percall  cumtime  percall
    50000   15.000    0.000   15.000    0.000 compute_reward()
```
- High tottime AND high cumtime
- The function itself is doing expensive work
- **Action:** Optimize the function's algorithm

### Pattern 2: Delegation Function
```
   ncalls  tottime  percall  cumtime  percall
      100    0.001    0.000   30.000    0.300 train_epoch()
    10000    0.500    0.000   29.000    0.003 train_step()
```
- Low tottime, high cumtime
- Time is spent in subfunctions
- **Action:** Look at what it calls

### Pattern 3: Death by 1000 Cuts
```
   ncalls  tottime  percall  cumtime  percall
 1000000    8.000    0.000    8.000    0.000 small_func()
```
- Very high ncalls
- Low per-call time but adds up
- **Action:** Reduce call count or inline

### Pattern 4: Recursive Functions
```
   ncalls  tottime  percall  cumtime  percall
  100/10    2.000    0.020    5.000    0.500 recursive_func()
```
- `100/10` means 100 total calls, 10 primitive (non-recursive)
- **Action:** Consider iterative alternative

## Sorting Options

| Sort Key | Use When |
|----------|----------|
| `cumulative` | Finding overall bottlenecks (default) |
| `time` | Finding functions that are slow themselves |
| `calls` | Finding frequently called functions |

## Common Pitfalls

### 1. Profiler Overhead
Profiling adds ~10-30% overhead. Relative times are still valid.

### 2. Missing Functions
Built-in functions (like `len`, `list.append`) may not appear. They're in "built-in" entries.

### 3. Generator Functions
Generators show time when iterated, not when defined.

### 4. Async Code
`async` functions may show misleading times due to event loop.

## Quick Analysis Workflow

1. **Sort by cumulative time** - Find top 5 bottlenecks
2. **Check tottime** - Is the function slow itself or delegating?
3. **Check ncalls** - Is it called too often?
4. **Drill down** - Look at what bottleneck functions call
5. **Measure again** - After optimization, re-profile

## Example Analysis

```
Profile output:
   ncalls  tottime  percall  cumtime  percall
      500    0.050    0.000   45.000    0.090 run_episode()
    25000    5.000    0.000   40.000    0.002 step()
   125000    2.000    0.000   30.000    0.000 forward()
   125000   25.000    0.000   25.000    0.000 {method 'matmul'}
```

**Analysis:**
1. `run_episode` has 45s cumtime but only 0.05s tottime → Look at children
2. `step` has 40s cumtime, 5s tottime → 35s in children, 5s own work
3. `forward` has 30s cumtime, 2s tottime → 28s in children
4. `matmul` has 25s tottime = 25s cumtime → This is the actual work

**Conclusion:** Matrix multiplication is the real bottleneck.

**Actions:**
- Reduce matrix sizes if possible
- Batch operations to reduce overhead
- Consider GPU acceleration
- Use more efficient BLAS library
