import subprocess
import sys
from pathlib import Path

# Root folder of the repository
REPO_ROOT = Path(__file__).resolve().parents[4]  # Go up to lunar-lander-file-folder

# Path to the specific virtual environment Python executable
VENV_PYTHON = REPO_ROOT / "lunar-lander" / ".venv-3.12.5" / "Scripts" / "python.exe"

# Path to the resume script
RESUME_SCRIPT = REPO_ROOT / "lunar-lander" / "tools" / "resume_exp023.py"

# Working directory for simulation
LUNAR_LANDER_DIR = REPO_ROOT / "lunar-lander"

def main():
    # Clear __pycache__ to ensure fresh code is used
    import shutil
    for cache_dir in LUNAR_LANDER_DIR.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Run the resume script from the lunar-lander directory
    subprocess.check_call([str(VENV_PYTHON), str(RESUME_SCRIPT)], cwd=str(LUNAR_LANDER_DIR))

if __name__ == "__main__":
    main()
