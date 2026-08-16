"""
One-Click Launcher for Drift-Sense DRAM Metrology Explorer
==========================================================
Runs the backend API, serves the modern glassmorphic frontend,
and automatically opens your default web browser to the dashboard.

Usage:
    python run.py
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def check_dependencies():
    """Verifies all required core packages are installed."""
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "numpy": "numpy",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "scipy": "scipy",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"\n[ERROR] Missing required Python package(s): {', '.join(missing)}")
        print("Please install them using:")
        print(f"    pip install -r requirements.txt\n")
        sys.exit(1)


def open_browser(url: str, delay_sec: float = 1.2):
    """Opens the local dashboard in the default web browser after server initializes."""
    time.sleep(delay_sec)
    print(f"\n🚀 Opening browser to: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[Notice] Could not auto-launch browser ({e}). Please visit {url} manually.")


def main():
    print("=" * 70)
    print("🔬 DRIFT-SENSE & FINFET SEMICONDUCTOR EXPLORER")
    print("=" * 70)
    print("[1/3] Checking environment & packages...")
    check_dependencies()
    print("      ✓ Core dependencies verified.")

    host = "127.0.0.1"
    port = 8000
    dashboard_url = f"http://localhost:{port}"

    print(f"[2/3] Preparing dashboard at {dashboard_url}...")
    # Launch browser thread
    threading.Thread(target=open_browser, args=(dashboard_url,), daemon=True).start()

    print("[3/3] Starting async FastAPI backend & frontend server...")
    print("      Press Ctrl+C at any time to shut down the server.\n")

    import uvicorn
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
