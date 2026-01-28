---
name: simulation-execution
description: Run the lunar lander simulation or training. Use when user asks to run, execute, start, or re-run the simulation, training, or main.py. Also use when deciding to test changes by running the simulation.
allowed-tools: Bash, Read
---

# Simulation Execution Protocol

## STATUS
**MANDATORY** - This skill MUST be followed for all simulation runs.

## RULE
Whenever you want to run or re-run the simulation, you MUST use the wrapper:
`scripts/run_simulation.py`

**Do NOT run `main.py` directly.**

## USAGE

### Starting the simulation
To run the simulation, execute:
```bash
"C:\Repositories for Git\lunar-lander-file-folder\lunar-lander\.venv-3.12.5\Scripts\python.exe" "C:\Repositories for Git\lunar-lander-file-folder\.claude\skills\simulation-execution\scripts\run_simulation.py"
```

### Stopping the simulation
To stop a running simulation gracefully, execute in a separate terminal:
```bash
"C:\Repositories for Git\lunar-lander-file-folder\lunar-lander\.venv-3.12.5\Scripts\python.exe" "C:\Repositories for Git\lunar-lander-file-folder\.claude\skills\simulation-execution\scripts\stop_simulation.py"
```

Options:
- `--no-wait` - Send stop signal without waiting for acknowledgment
- `--clear` - Clear any existing stop signal (use before starting a new run)
- `--timeout N` - Wait up to N seconds for simulation to stop (default: 30)

## RATIONALE
The simulation produces high-volume diagnostic output that can exhaust an AI agent's context window and degrade long-horizon reasoning.

The wrapper enforces an agent-safe execution environment by:
- Automatically setting print_mode to 'background' (minimal output)
- Only printing batch completion summaries (e.g., "Batch 1 (Runs 1-100) completed. Success: 45%")
- Preventing verbose per-episode diagnostics
- Preserving agent context for interpreting trends, diagnostics, and charts

## ENFORCEMENT
- Do not bypass the wrapper
- Do not manually modify configuration
- Do not assume the environment is already safe
- Do not run `src/main.py` directly

**Any simulation run that does not go through `run_simulation.py` is considered a protocol violation.**
