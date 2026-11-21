"""Run every part of the NVDA November 2026 case study.

Usage (from the repo root)::

    uv run python case_studies/nvda_2026/run_all.py

Figures and tables are written to ``case_studies/nvda_2026/results/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling scripts importable however this file is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import event_study  # noqa: E402
import models_on_nvda  # noqa: E402


def main() -> None:
    """Phase 1 (event study), then Phase 2 (the four models on NVDA)."""
    event_study.main()
    models_on_nvda.main()


if __name__ == "__main__":
    main()
