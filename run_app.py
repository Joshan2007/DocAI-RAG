"""
DocAI — One-Command Launcher
Run this script once. It handles everything automatically:
  • Creates a Python virtual environment (.venv) if it doesn't exist
  • Installs all Python dependencies (requirements.txt)
  • Installs all Node dependencies (frontend/node_modules)
  • Prompts for your Gemini API key (one time only, saved to .env)
  • Starts the FastAPI backend on http://127.0.0.1:8000
  • Starts the Next.js frontend on http://localhost:3000

Usage:
    python run_app.py
"""

import sys
import os
import time
import subprocess
import signal
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
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

# ── Detect the right Python / pip / npm executables ───────────────────────────
IS_WIN = os.name == "nt"

VENV_PYTHON = VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")
VENV_PIP    = VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("pip.exe"    if IS_WIN else "pip")
NPM_CMD     = "npm.cmd" if IS_WIN else "npm"

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

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Python dependencies
# ══════════════════════════════════════════════════════════════════════════════
def ensure_python_deps():
    req_file = ROOT_DIR / "requirements.txt"
    if not req_file.exists():
        print(red("  ✘ requirements.txt not found — skipping pip install."))
        return
    # Sentinel file avoids re-installing on every launch
    sentinel = VENV_DIR / ".deps_installed"
    req_mtime = req_file.stat().st_mtime
    if sentinel.exists() and sentinel.stat().st_mtime >= req_mtime:
        print(green("  ✔ Python dependencies already installed."))
        return
    print(yellow("  Installing Python dependencies (this may take a minute)…"))
    run([str(VENV_PIP), "install", "-r", str(req_file)])
    sentinel.touch()
    print(green("  ✔ Python dependencies installed."))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Node dependencies
# ══════════════════════════════════════════════════════════════════════════════
def ensure_node_deps():
    node_modules = FRONTEND_DIR / "node_modules"
    pkg_json     = FRONTEND_DIR / "package.json"
    sentinel     = FRONTEND_DIR / ".npm_installed"

    if not pkg_json.exists():
        print(red("  ✘ frontend/package.json not found — skipping npm install."))
        return

    pkg_mtime = pkg_json.stat().st_mtime
    if sentinel.exists() and sentinel.stat().st_mtime >= pkg_mtime and node_modules.exists():
        print(green("  ✔ Node dependencies already installed."))
        return

    print(yellow("  Installing Node dependencies (this may take a minute)…"))
    run([NPM_CMD, "install"], cwd=FRONTEND_DIR)
    sentinel.touch()
    print(green("  ✔ Node dependencies installed."))

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
# PHASE 5 — Launch servers
# ══════════════════════════════════════════════════════════════════════════════
def start_servers():
    print(yellow("  Starting FastAPI backend on http://127.0.0.1:8000 …"))
    backend_proc = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "uvicorn", "backend.server:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print(yellow("  Starting Next.js frontend on http://localhost:3000 …"))
    next_built = (FRONTEND_DIR / ".next").exists()
    frontend_script = "start" if next_built else "dev"
    frontend_proc = subprocess.Popen(
        [NPM_CMD, "run", frontend_script],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    time.sleep(4)

    print()
    print(bold("=" * 65))
    print(bold("  🚀 DocAI is LIVE!"))
    print(f"  {green('Frontend')} → http://localhost:3000")
    print(f"  {green('Backend API')} → http://127.0.0.1:8000/docs")
    print(bold("=" * 65))
    print(cyan("  Press Ctrl+C to stop both servers.\n"))

    def cleanup(*_):
        print(yellow("\n  Shutting down servers…"))
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()

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

    step(3, 5, "Checking Node.js dependencies…")
    ensure_node_deps()

    step(4, 5, "Checking Gemini API key…")
    ensure_api_key()

    step(5, 5, "Launching servers…")
    start_servers()
