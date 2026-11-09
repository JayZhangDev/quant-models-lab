"""Shared helpers: data loading, plotting style and performance metrics."""

from pathlib import Path

# This file lives at <repo>/src/qml/common/__init__.py, so the repo root is three folders up.
# Every other module builds its paths from here, so scripts work from any working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
