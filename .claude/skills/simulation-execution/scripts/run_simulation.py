import subprocess
import sys
from pathlib import Path

# Root folder of the repository
REPO_ROOT = Path(__file__).resolve().parents[4]  # Go up to lunar-lander-file-folder

# Path to the specific virtual environment Python executable
VENV_PYTHON = REPO_ROOT / "lunar-lander" / ".venv-3.12.5" / "Scripts" / "python.exe"

# Path to the background print-mode script
SET_BACKGROUND_SCRIPT = REPO_ROOT / ".claude" / "skills" / "simulation-execution" / "scripts" / "set_background_print_mode.py"

# Path to main.py
MAIN_SCRIPT = REPO_ROOT / "lunar-lander" / "src" / "main.py"

# Working directory for simulation
LUNAR_LANDER_DIR = REPO_ROOT / "lunar-lander"

def main():
    # Step 1: Ensure background print mode (minimal output: batch completions only)
    subprocess.check_call([str(VENV_PYTHON), str(SET_BACKGROUND_SCRIPT)])

    # Step 2: Run the main simulation from the lunar-lander directory
    subprocess.check_call([str(VENV_PYTHON), str(MAIN_SCRIPT)], cwd=str(LUNAR_LANDER_DIR))

if __name__ == "__main__":
    main()
