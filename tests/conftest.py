from __future__ import annotations

from pathlib import Path

import pytest

from stablediff.util import suppress_known_upstream_warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET_DIR = REPO_ROOT / "data" / "sample"


@pytest.fixture(autouse=True)
def _quiet_upstream_deprecations() -> None:
    suppress_known_upstream_warnings()


@pytest.fixture
def sample_image_dir() -> Path:
    """Bundled tiny image set committed in the repo (``data/sample``)."""
    if not SAMPLE_DATASET_DIR.is_dir():
        pytest.skip(f"Bundled sample dataset not found: {SAMPLE_DATASET_DIR}")

    images = sorted(SAMPLE_DATASET_DIR.glob("*.jpg"))
    if not images:
        pytest.skip(f"No images found under {SAMPLE_DATASET_DIR}")
    return SAMPLE_DATASET_DIR
