"""Force Qt offscreen so importing the window never needs a display."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
