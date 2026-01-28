# Slow vs Fast Python Patterns

## Quick Reference Table

| Slow Pattern | Fast Alternative | Speedup |
|--------------|------------------|---------|
| `for` + `.append()` | List comprehension | 1.5-2x |
| String `+=` in loop | `''.join(list)` | 10-100x |
| `x in list` | `x in set` | O(n) → O(1) |
| `range(len(x))` | `enumerate(x)` | ~same, cleaner |
| `dict.keys()` membership | `key in dict` | ~same, cleaner |
| Multiple `dict[key]` | Cache in variable | 1.2-1.5x |
| Nested loops | `itertools.product` | 1.1-1.3x |
| `sorted()[0]` | `min()` | O(n log n) → O(n) |
| `sorted()[-1]` | `max()` | O(n log n) → O(n) |

---

## Detailed Patterns

### 1. List Building in Loops

**Slow:**
```python
results = []
for x in data:
    results.append(process(x))
```

**Fast:**
```python
results = [process(x) for x in data]
```

**Why:** List comprehensions are optimized at the bytecode level. The append method lookup and call overhead is eliminated.

**When to use slow version:** When you need `try/except` inside the loop or complex multi-statement logic.

---

### 2. String Concatenation

**Slow:**
```python
result = ""
for word in words:
    result += word + " "
```

**Fast:**
```python
result = " ".join(words)
```

**Why:** Strings are immutable. Each `+=` creates a new string object and copies all previous content. `join()` pre-allocates the exact size needed.

**Speedup:** 10-100x for many iterations (quadratic vs linear).

---

### 3. Membership Testing

**Slow:**
```python
if item in my_list:  # O(n)
    ...
```

**Fast:**
```python
my_set = set(my_list)  # One-time O(n) conversion
if item in my_set:     # O(1)
    ...
```

**When to convert:** If you check membership more than once, convert to set.

**Also applies to:**
```python
# Slow
if x in [1, 2, 3, 4, 5]:

# Fast
if x in {1, 2, 3, 4, 5}:
```

---

### 4. Dictionary Access

**Slow:**
```python
for i in range(1000):
    value = config['setting']['subsetting']['value']
    process(value)
```

**Fast:**
```python
value = config['setting']['subsetting']['value']
for i in range(1000):
    process(value)
```

**Why:** Each `[]` is a method call and hash lookup. Cache values accessed repeatedly.

---

### 5. Finding Min/Max

**Slow:**
```python
smallest = sorted(data)[0]
largest = sorted(data)[-1]
```

**Fast:**
```python
smallest = min(data)
largest = max(data)
```

**Why:** `sorted()` is O(n log n), `min()`/`max()` are O(n).

**For multiple values:**
```python
# Need both min and max
import heapq
smallest = heapq.nsmallest(1, data)[0]
largest = heapq.nlargest(1, data)[0]

# Or single pass
smallest, largest = min(data), max(data)  # Two passes but still O(n)
```

---

### 6. Checking Empty Collections

**Slow:**
```python
if len(my_list) == 0:
if len(my_dict) == 0:
```

**Fast:**
```python
if not my_list:
if not my_dict:
```

**Why:** Empty collections are falsy. Direct boolean check is faster than `len()` call.

---

### 7. Counting Occurrences

**Slow:**
```python
counts = {}
for item in data:
    if item in counts:
        counts[item] += 1
    else:
        counts[item] = 1
```

**Fast:**
```python
from collections import Counter
counts = Counter(data)
```

**Alternative:**
```python
from collections import defaultdict
counts = defaultdict(int)
for item in data:
    counts[item] += 1
```

---

### 8. Iterating with Index

**Slow:**
```python
for i in range(len(items)):
    item = items[i]
    process(i, item)
```

**Fast:**
```python
for i, item in enumerate(items):
    process(i, item)
```

**Why:** Avoids repeated indexing. More Pythonic and slightly faster.

---

### 9. Multiple Conditions

**Slow:**
```python
if x == 1 or x == 2 or x == 3 or x == 4:
```

**Fast:**
```python
if x in {1, 2, 3, 4}:
```

**Why:** Set lookup is O(1). Chain of `or` evaluates each condition.

---

### 10. Default Dictionary Values

**Slow:**
```python
if key in d:
    value = d[key]
else:
    value = default
```

**Fast:**
```python
value = d.get(key, default)
```

**For mutation:**
```python
# Slow
if key not in d:
    d[key] = []
d[key].append(item)

# Fast
d.setdefault(key, []).append(item)

# Fastest (for repeated use)
from collections import defaultdict
d = defaultdict(list)
d[key].append(item)
```

---

## When NOT to Optimize

1. **Readability matters more** for code run rarely
2. **Premature optimization** before profiling
3. **Micro-optimizations** that save nanoseconds
4. **One-time setup code** that runs once at startup

**Rule:** Profile first, optimize the actual bottlenecks.
