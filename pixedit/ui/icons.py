"""Toolbar icons. SVGs live in ``pixedit/resources/icons`` and match the dark chrome."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QIcon

_ICONS = Path(__file__).resolve().parents[1] / "resources" / "icons"

# Plugin tool ids are namespaced. The file stays a short word.
_ALIASES = {"example.marker": "marker"}


def icon_for_tool(tool_id: str) -> QIcon:
    name = _ALIASES.get(tool_id, tool_id)
    path = _ICONS / f"{name}.svg"
    if not path.is_file():
        return QIcon()
    return QIcon(str(path))
