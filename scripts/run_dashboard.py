"""
CLI Launcher Script for E-Commerce Intelligence Dashboard.

Usage:
    python scripts/run_dashboard.py [--port 8501]
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "src" / "dashboard" / "app.py"


def main():
    parser = argparse.ArgumentParser(description="Run E-Commerce Intelligence Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--server.headless",
        "true",
        "--theme.base",
        "light",
    ]
    print(f"Launching Streamlit on http://{args.host}:{args.port}...")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
