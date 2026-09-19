"""
DocAI — One-Command Launcher
Run this script once. It handles everything automatically:
  • Creates a Python virtual environment (.venv) if it doesn't exist
  • Installs all Python dependencies (requirements.txt)
  • Prompts for your Gemini API key (one time only, saved to .env)
    • Starts the Streamlit app on http://localhost:8501

Usage:
    python run_app.py
"""

import sys
import os
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
ENV_FILE = ROOT_DIR / ".env"

# ── Colour helpers (no deps needed) ───────────────────────────────────────────
def _c(code, text):
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def red(t):    return _c("31", t)
def bold(t):   return _c("1",  t)

# ── Detect the right Python / pip executable ─────────────────────────────────
IS_WIN = os.name == "nt"

VENV_PYTHON = VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")
VENV_PIP    = VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("pip.exe"    if IS_WIN else "pip")

# ── Utilities ──────────────────────────────────────────────────────────────────
def run(cmd, cwd=None, check=True):
    """Run a command, streaming its output in real time."""
    subprocess.run(cmd, cwd=str(cwd or ROOT_DIR), check=check)

def step(n, total, msg):
    print(f"\n{bold(f'[{n}/{total}]')} {cyan(msg)}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Python virtual environment
# ══════════════════════════════════════════════════════════════════════════════
def ensure_venv():
    if VENV_PYTHON.exists():
        print(green("  ✔ Python virtual environment already exists."))
        return
    print(yellow("  Creating Python virtual environment (.venv)…"))
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    print(green("  ✔ Virtual environment created."))

def start_streamlit():
    print(yellow("  Starting Streamlit app on http://localhost:8501 …"))
    run([str(VENV_PYTHON), "-m", "streamlit", "run", "app.py"], cwd=ROOT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Gemini API key
# ══════════════════════════════════════════════════════════════════════════════
def ensure_api_key():
    """Read .env; if GEMINI_API_KEY is missing or a placeholder, prompt the user."""
    # Check shell environment first
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key and env_key != "your_gemini_api_key_here":
        print(green("  ✔ GEMINI_API_KEY found in environment."))
        return

    # Parse existing .env
    env_vars = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip()

    key = env_vars.get("GEMINI_API_KEY", "")
    if key and key != "your_gemini_api_key_here":
        print(green("  ✔ GEMINI_API_KEY found in .env."))
        os.environ["GEMINI_API_KEY"] = key
        return

    # Prompt the user
    print()
    print(bold("  🔑  Gemini API Key required"))
    print("  Get a free key at: " + cyan("https://aistudio.google.com/app/apikey"))
    print()
    while True:
        try:
            key = input("  Paste your Gemini API key and press Enter: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(red("\n  Aborted."))
            sys.exit(1)
        if key and len(key) > 10 and "your_" not in key:
            break
        print(yellow("  That doesn't look like a valid key. Please try again."))

    # Write / update .env
    env_vars["GEMINI_API_KEY"] = key
    if "DEFAULT_MODEL" not in env_vars:
        env_vars["DEFAULT_MODEL"] = "gemini-2.0-flash"

    lines = []
    if ENV_FILE.exists():
        raw = ENV_FILE.read_text(encoding="utf-8").splitlines()
        replaced = set()
        for line in raw:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k = stripped.split("=", 1)[0].strip()
                if k in env_vars:
                    lines.append(f"{k}={env_vars[k]}")
                    replaced.add(k)
                    continue
            lines.append(line)
        for k, v in env_vars.items():
            if k not in replaced:
                lines.append(f"{k}={v}")
    else:
        for k, v in env_vars.items():
            lines.append(f"{k}={v}")

    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = key
    print(green("  ✔ API key saved to .env — you won't be asked again."))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Launch Streamlit
# ══════════════════════════════════════════════════════════════════════════════
def start_servers():
    start_streamlit()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print(bold("=" * 65))
    print(bold("  🧠 DocAI — AI Knowledge Assistant"))
    print(bold("=" * 65))

    step(1, 5, "Checking Python virtual environment…")
    ensure_venv()

    step(2, 5, "Checking Python dependencies…")
    ensure_python_deps()

    step(3, 4, "Checking Gemini API key…")
    ensure_api_key()

    step(4, 4, "Launching Streamlit…")
    start_servers()
