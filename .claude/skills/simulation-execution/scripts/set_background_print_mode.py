from pathlib import Path
import re

# Navigate from .claude/skills/simulation-execution/scripts/ to lunar-lander/src/config.py
REPO_ROOT = Path(__file__).resolve().parents[4]  # Go up to lunar-lander-file-folder
CONFIG_PATH = REPO_ROOT / "lunar-lander" / "src" / "config.py"

def set_print_mode_background():
    """Set print_mode to 'background' if it's anything other than 'background'."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.py not found at {CONFIG_PATH}")

    text = CONFIG_PATH.read_text()

    # Match dataclass field syntax: print_mode: str = 'value'
    pattern = re.compile(
        r"(print_mode:\s*str\s*=\s*)(['\"])(.*?)\2",
        re.MULTILINE,
    )

    match = pattern.search(text)
    if not match:
        raise RuntimeError("print_mode not found inside RunConfig")

    current_mode = match.group(3)

    # Only change if not already 'background'
    if current_mode == 'background':
        print("print_mode already set to 'background'")
        return

    def replacer(m):
        return f"{m.group(1)}'background'"

    new_text, count = pattern.subn(replacer, text, count=1)

    CONFIG_PATH.write_text(new_text)
    print(f"print_mode changed from '{current_mode}' to 'background'")

if __name__ == "__main__":
    set_print_mode_background()
