"""Stop the running simulation gracefully.

Creates a stop signal file that the simulation checks for and exits cleanly.
"""

from pathlib import Path
import time

# Stop signal file location (in lunar-lander directory)
REPO_ROOT = Path(__file__).resolve().parents[4]

# Path to the specific virtual environment Python executable
VENV_PYTHON = REPO_ROOT / "lunar-lander" / ".venv-3.12.5" / "Scripts" / "python.exe"

STOP_FILE = REPO_ROOT / "lunar-lander" / ".stop_simulation"


def stop_simulation(wait: bool = True, timeout: float = 30.0) -> bool:
    """Signal the simulation to stop.

    Args:
        wait: If True, wait for the simulation to acknowledge and clean up
        timeout: Maximum seconds to wait for simulation to stop

    Returns:
        True if stop signal was sent (and acknowledged if wait=True)
    """
    # Create the stop signal file
    STOP_FILE.write_text("stop")
    print(f"Stop signal sent: {STOP_FILE}")

    if not wait:
        return True

    # Wait for simulation to remove the file (acknowledging stop)
    start = time.time()
    while STOP_FILE.exists():
        if time.time() - start > timeout:
            print(f"Timeout waiting for simulation to stop after {timeout}s")
            # Clean up the file anyway
            STOP_FILE.unlink(missing_ok=True)
            return False
        time.sleep(0.5)

    print("Simulation stopped successfully")
    return True


def clear_stop_signal():
    """Clear any existing stop signal (use before starting a new simulation)."""
    if STOP_FILE.exists():
        STOP_FILE.unlink()
        print("Cleared existing stop signal")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stop the running simulation")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for acknowledgment")
    parser.add_argument("--clear", action="store_true", help="Clear stop signal instead of sending")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds")
    args = parser.parse_args()

    if args.clear:
        clear_stop_signal()
    else:
        stop_simulation(wait=not args.no_wait, timeout=args.timeout)
