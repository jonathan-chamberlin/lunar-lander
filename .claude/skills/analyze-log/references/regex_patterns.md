# Regex Patterns for Log Parsing

## Basic Episode Pattern

```python
import re

# Full episode line
EPISODE_PATTERN = r"Run (\d+)\s*([✓✗])\s*(\w+).*Reward:\s*([\d.-]+)"

# Usage
match = re.search(EPISODE_PATTERN, line)
if match:
    episode = int(match.group(1))
    success = match.group(2) == "✓"
    outcome = match.group(3)
    reward = float(match.group(4))
```

## Extended Patterns

### With Reward Breakdown

```python
# Capture env and shaped rewards
REWARD_BREAKDOWN = r"Reward:\s*([\d.-]+)\s*\(env:\s*([\d.-]+)\s*/\s*shaped:\s*([+\d.-]+)\)"

match = re.search(REWARD_BREAKDOWN, line)
if match:
    total_reward = float(match.group(1))
    env_reward = float(match.group(2))
    shaped_reward = float(match.group(3))
```

### Success Detection

```python
# Multiple ways to detect success
SUCCESS_PATTERNS = [
    r"✓",              # Check mark
    r"✅",             # Green check emoji
    r"LANDED",         # Outcome contains LANDED
    r"Landed Safely",  # Message
]

def is_success(line: str) -> bool:
    return any(re.search(p, line) for p in SUCCESS_PATTERNS)
```

### Outcome Extraction

```python
# Known outcomes
OUTCOMES = [
    "LANDED_PERFECTLY",
    "LANDED_SOFTLY",
    "LANDED_HARD",
    "CRASHED_HIGH_VELOCITY",
    "CRASHED_FAST_HORIZONTAL",
    "CRASHED_TILTED",
    "FLEW_OFF_LEFT",
    "FLEW_OFF_RIGHT",
    "FLEW_OFF_TOP",
    "TIMEOUT",
]

OUTCOME_PATTERN = r"\b(" + "|".join(OUTCOMES) + r")\b"

match = re.search(OUTCOME_PATTERN, line)
if match:
    outcome = match.group(1)
```

## Flexible Patterns

### Reward Only (Any Format)

```python
# Matches various reward formats
REWARD_PATTERNS = [
    r"Reward:\s*([\d.-]+)",
    r"reward\s*=\s*([\d.-]+)",
    r"R:\s*([\d.-]+)",
    r"total:\s*([\d.-]+)",
]

def extract_reward(line: str) -> float | None:
    for pattern in REWARD_PATTERNS:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None
```

### Episode Number

```python
# Various episode number formats
EPISODE_PATTERNS = [
    r"Run (\d+)",
    r"Episode (\d+)",
    r"Ep\.?\s*(\d+)",
    r"#(\d+)",
]

def extract_episode(line: str) -> int | None:
    for pattern in EPISODE_PATTERNS:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None
```

## Complete Parser

```python
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class EpisodeResult:
    episode: int
    success: bool
    outcome: str
    reward: float
    env_reward: Optional[float] = None
    shaped_reward: Optional[float] = None


def parse_episode_line(line: str) -> Optional[EpisodeResult]:
    """Parse a single episode result line."""

    # Try standard format first
    standard = r"Run (\d+)\s*([✓✗])\s*(\w+).*Reward:\s*([\d.-]+)"
    match = re.search(standard, line)

    if not match:
        return None

    result = EpisodeResult(
        episode=int(match.group(1)),
        success=match.group(2) == "✓",
        outcome=match.group(3),
        reward=float(match.group(4))
    )

    # Try to extract breakdown
    breakdown = r"env:\s*([\d.-]+)\s*/\s*shaped:\s*([+\d.-]+)"
    breakdown_match = re.search(breakdown, line)
    if breakdown_match:
        result.env_reward = float(breakdown_match.group(1))
        result.shaped_reward = float(breakdown_match.group(2))

    return result


def parse_log_file(filepath: str) -> list[EpisodeResult]:
    """Parse all episodes from a log file."""
    results = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            result = parse_episode_line(line)
            if result:
                results.append(result)

    return results
```

## Handling Edge Cases

### Unicode Issues

```python
# Handle various encodings
def safe_read(filepath: str) -> str:
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    # Fallback with error handling
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()
```

### Missing Fields

```python
def safe_float(s: str, default: float = 0.0) -> float:
    """Safely convert string to float."""
    try:
        return float(s.replace('+', ''))
    except (ValueError, AttributeError):
        return default
```

### Multiline Entries

```python
def parse_multiline_entry(lines: list[str], start_idx: int) -> tuple[dict, int]:
    """Parse entry that may span multiple lines."""
    entry = {}
    i = start_idx

    # Collect lines until next entry or end
    while i < len(lines):
        line = lines[i]

        # Check if this is a new entry
        if re.match(r"Run \d+", line) and i > start_idx:
            break

        # Extract data from this line
        # ... parsing logic ...

        i += 1

    return entry, i
```

## Performance Tips

1. **Compile patterns** for repeated use:
   ```python
   EPISODE_RE = re.compile(r"Run (\d+)\s*([✓✗])\s*(\w+).*Reward:\s*([\d.-]+)")
   ```

2. **Use `re.search` not `re.match`** unless matching from start

3. **Early exit** when possible:
   ```python
   if "Run" not in line:
       continue
   ```

4. **Batch process** large files:
   ```python
   def parse_in_chunks(filepath: str, chunk_size: int = 10000):
       with open(filepath) as f:
           while True:
               lines = list(itertools.islice(f, chunk_size))
               if not lines:
                   break
               yield from (parse_line(l) for l in lines)
   ```
