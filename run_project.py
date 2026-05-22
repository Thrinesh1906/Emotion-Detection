"""
One-click project runner: setup, train, evaluate, launch dashboard.
Usage: python run_project.py [--train] [--evaluate] [--app] [--api]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Emotion Transition Detection project")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--train", action="store_true", help="Train BiLSTM model")
    parser.add_argument("--quick", action="store_true", help="Quick training mode")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI server")
    parser.add_argument("--all", action="store_true", help="Install, train, evaluate, launch app")
    args = parser.parse_args()

    if args.all:
        args.install = args.train = args.evaluate = args.app = True
        args.quick = True

    if args.install:
        run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if args.train:
        cmd = [sys.executable, "-m", "training.train"]
        if args.quick:
            cmd.append("--quick")
        run_cmd(cmd)

    if args.evaluate:
        run_cmd([sys.executable, "-m", "training.evaluate"])

    if args.api:
        run_cmd([sys.executable, "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"])

    if args.app:
        run_cmd([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

    if not any([args.install, args.train, args.evaluate, args.app, args.api, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
