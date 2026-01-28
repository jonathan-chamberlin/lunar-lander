# NumPy Vectorization Patterns

## Why Vectorize?

Python loops over NumPy arrays are **10-100x slower** than vectorized operations because:
- Python loop overhead (bytecode interpretation)
- Type checking on every element
- NumPy operations use optimized C/Fortran code

---

## Common Conversions

### 1. Element-wise Operations

**Slow (Python loop):**
```python
result = np.zeros(len(arr))
for i in range(len(arr)):
    result[i] = arr[i] * 2 + 1
```

**Fast (Vectorized):**
```python
result = arr * 2 + 1
```

---

### 2. Conditional Operations

**Slow:**
```python
result = np.zeros(len(arr))
for i in range(len(arr)):
    if arr[i] > 0:
        result[i] = arr[i]
    else:
        result[i] = 0
```

**Fast:**
```python
result = np.where(arr > 0, arr, 0)
# Or
result = np.maximum(arr, 0)  # ReLU
# Or
result = arr.clip(min=0)
```

---

### 3. Aggregations

**Slow:**
```python
total = 0
for x in arr:
    total += x
mean = total / len(arr)
```

**Fast:**
```python
total = arr.sum()
mean = arr.mean()
# Also: arr.std(), arr.var(), arr.min(), arr.max()
```

---

### 4. Finding Indices

**Slow:**
```python
indices = []
for i in range(len(arr)):
    if arr[i] > threshold:
        indices.append(i)
```

**Fast:**
```python
indices = np.where(arr > threshold)[0]
# Or for boolean mask
mask = arr > threshold
```

---

### 5. Applying Functions

**Slow:**
```python
result = np.zeros(len(arr))
for i in range(len(arr)):
    result[i] = custom_func(arr[i])
```

**Medium (np.vectorize - still slow):**
```python
vfunc = np.vectorize(custom_func)
result = vfunc(arr)
```

**Fast (if possible, use numpy operations):**
```python
# Instead of custom_func that does: x**2 + 2*x + 1
result = arr**2 + 2*arr + 1
```

**Note:** `np.vectorize` is just a convenience wrapper, not true vectorization.

---

### 6. Matrix Operations

**Slow:**
```python
result = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        result[i, j] = A[i, :] @ B[:, j]
```

**Fast:**
```python
result = A @ B  # Matrix multiplication
# Or
result = np.dot(A, B)
result = np.matmul(A, B)
```

---

### 7. Broadcasting

**Slow:**
```python
# Add bias to each row
result = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    result[i] = matrix[i] + bias
```

**Fast:**
```python
result = matrix + bias  # Broadcasting handles it
```

**Broadcasting rules:**
- Dimensions are compared right-to-left
- Dimensions match if equal or one is 1
- Missing dimensions are treated as 1

```python
# Examples
(3, 4) + (4,)    -> (3, 4)  # bias added to each row
(3, 4) + (3, 1)  -> (3, 4)  # bias added to each column
(3, 1, 4) + (1, 5, 4) -> (3, 5, 4)
```

---

### 8. Batch Processing

**Slow:**
```python
results = []
for sample in batch:
    results.append(process_single(sample))
result = np.array(results)
```

**Fast:**
```python
# Reshape batch to process all at once
result = process_batch(batch)  # Vectorized batch operation
```

---

### 9. Distance Calculations

**Slow:**
```python
distances = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        distances[i, j] = np.sqrt(np.sum((points[i] - points[j])**2))
```

**Fast:**
```python
from scipy.spatial.distance import cdist
distances = cdist(points, points)

# Or with numpy only
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt(np.sum(diff**2, axis=-1))
```

---

### 10. Normalization

**Slow:**
```python
for i in range(arr.shape[0]):
    arr[i] = (arr[i] - arr[i].mean()) / arr[i].std()
```

**Fast:**
```python
# Normalize each row
mean = arr.mean(axis=1, keepdims=True)
std = arr.std(axis=1, keepdims=True)
arr_normalized = (arr - mean) / std
```

---

## RL-Specific Patterns

### Batch Advantage Calculation

**Slow:**
```python
advantages = np.zeros(len(rewards))
for i in range(len(rewards)):
    advantages[i] = rewards[i] - baseline[i]
```

**Fast:**
```python
advantages = rewards - baseline
```

### Discount Cumsum (GAE)

**Slow:**
```python
returns = np.zeros(len(rewards))
running = 0
for i in reversed(range(len(rewards))):
    running = rewards[i] + gamma * running
    returns[i] = running
```

**Fast:**
```python
from scipy.signal import lfilter
returns = lfilter([1], [1, -gamma], rewards[::-1])[::-1]
```

### Batch State Processing

**Slow:**
```python
q_values = []
for state in states:
    q_values.append(network(state))
```

**Fast:**
```python
q_values = network(np.stack(states))  # Single forward pass
```

---

## Memory Tips

1. **Avoid unnecessary copies:**
   ```python
   # Creates copy
   b = a.copy()

   # View (no copy)
   b = a.view()
   b = a.reshape(...)  # Usually a view
   ```

2. **In-place operations:**
   ```python
   # Creates new array
   a = a + 1

   # In-place
   a += 1
   np.add(a, 1, out=a)
   ```

3. **Pre-allocate:**
   ```python
   # Slow - reallocates
   result = np.array([])
   for x in data:
       result = np.append(result, x)

   # Fast - pre-allocated
   result = np.empty(len(data))
   for i, x in enumerate(data):
       result[i] = x

   # Fastest - vectorized
   result = np.array(data)
   ```
