#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"


def resolve_runs_path(path_like: str | Path) -> Path:
    """
    Normalize output paths to live under runs/ for relative paths.

    Rules:
    - absolute path: keep as-is
    - relative path starting with runs/: resolve under project root
    - other relative path: prefix with runs/
    """
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "runs":
        return ROOT / path
    return RUNS_ROOT / path
