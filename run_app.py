"""
Launcher script to run both DocAI FastAPI Backend (port 8000)
and Next.js Frontend (port 3000) locally.
"""

import sys
import os
import time
import subprocess
import signal
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

print("=" * 65)
print("🧠 DocAI — Production AI Knowledge Assistant Launcher")
print("=" * 65)

# 1. Start FastAPI Backend
print("[1/2] Starting FastAPI RAG Engine on http://127.0.0.1:8000...")
backend_cmd = [
    str(VENV_PYTHON), "-m", "uvicorn",
    "backend.server:app",
    "--host", "127.0.0.1",
    "--port", "8000"
]
backend_proc = subprocess.Popen(
    backend_cmd,
    cwd=str(ROOT_DIR),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# 2. Start Next.js Frontend
print("[2/2] Starting Next.js UI on http://localhost:3000...")
next_built = (FRONTEND_DIR / ".next").exists()
frontend_script = "start" if next_built else "dev"
frontend_cmd = ["npm.cmd" if os.name == "nt" else "npm", "run", frontend_script]
frontend_proc = subprocess.Popen(
    frontend_cmd,
    cwd=str(FRONTEND_DIR),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

time.sleep(3)
print("\n" + "*" * 65)
print("🚀 DocAI is LIVE!")
print("👉 Local Host Link: http://localhost:3000")
print("👉 Backend API Link: http://127.0.0.1:8000/docs")
print("*" * 65 + "\n")
print("Press Ctrl+C to stop both servers.\n")

def cleanup(*args):
    print("\nShutting down servers...")
    backend_proc.terminate()
    frontend_proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    cleanup()
