#!/usr/bin/env python
"""
train.py  –  Convenience entry-point at the project root.

Usage:
    python train.py

This just delegates to src.pipeline.run() so all logic stays in src/.
"""
import sys
from pathlib import Path

# Make sure `src` is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run

if __name__ == "__main__":
    run()
