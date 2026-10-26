"""Pointer shape shared by the mask and select tools."""

from __future__ import annotations


def _region_cursor(kind: str) -> str:
    """Pointer for a rectangle, ellipse, freehand loop, or magic wand."""
    if kind == "ellipse":
        return "select-ellipse"
    if kind == "freehand":
        return "lasso"
    if kind == "wand":
        return "wand"
    return "select-rect"
