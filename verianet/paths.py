"""Shared repository paths.

These constants make scripts runnable from any current working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "verysmallnn_weights.npz"
RESULTS_DIR = PROJECT_ROOT / "results"
COUNTERFACTUALS_DIR = RESULTS_DIR / "counterfactuals"
DEMO_DIR = PROJECT_ROOT / "demo"
PRESENTATION_DIR = PROJECT_ROOT / "presentation"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
