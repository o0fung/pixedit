"""Force Qt offscreen so importing the window never needs a display."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_output_config(tmp_path, monkeypatch):
    """Keep the output-folder setting out of the real app config."""
    monkeypatch.setenv("PIXEDIT_CONFIG", str(tmp_path / "pixedit.ini"))
