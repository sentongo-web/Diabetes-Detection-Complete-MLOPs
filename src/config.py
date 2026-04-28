"""
src/config.py
─────────────
Loads config/config.yaml once and exposes a typed Namespace so every module
can do  `from src.config import CFG`  instead of re-reading YAML every time.
"""

import yaml
from pathlib import Path
from types import SimpleNamespace

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def _dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


with open(_CONFIG_PATH, "r") as _f:
    _raw = yaml.safe_load(_f)

CFG = _dict_to_ns(_raw)
