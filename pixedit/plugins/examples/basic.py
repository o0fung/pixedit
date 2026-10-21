"""Shipped example plugin: a Vignette filter and a Marker tool.

Copy this file to ``~/.pixedit/plugins/`` and edit it, or add an entry point:

    [project.entry-points."pixedit.plugins"]
    my_plugin = "my_package.module:register"

``register(api)`` is the only required hook. Reload from the Plugins menu.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from pixedit.model import MutateLayerCommand, apply_visual, visual_state
from pixedit.tools.base import Tool


def vignette_image(image: Image.Image, strength: float = 0.65) -> Image.Image:
    """Darken pixels toward the corners. ``strength`` is 0..1."""
    arr = np.array(image.convert("RGBA")).astype(np.float32)
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    ny = (yy - cy) / max(cy, 1.0)
    nx = (xx - cx) / max(cx, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    falloff = np.clip((radius - 0.25) / 1.15, 0.0, 1.0)
    factor = 1.0 - float(strength) * falloff
    arr[..., :3] *= factor[..., None]
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGBA")


def apply_vignette(session) -> None:
    layer = session.active_layer()
    if layer is None or layer.kind == "redaction":
        session.show_message("Vignette needs a raster or text layer.")
        return
    before = visual_state(layer)
    image = vignette_image(layer.image, strength=0.7)
    layer.image = image
    layer.kind = "raster"
    layer.text = None
    new_state = visual_state(layer)
    apply_visual(layer, before)
    session.commit(MutateLayerCommand(layer, new_state, name="Vignette"))


class MarkerTool(Tool):
    """Stamp a plus sign. Demonstrates a plugin canvas tool."""

    id = "example.marker"
    label = "Marker"
    shortcut = ""
    cursor = "cross"
    hint = "Click to stamp a marker on the active layer."

    def option_schema(self):
        return [
            {
                "key": "marker_size",
                "label": "Marker size",
                "type": "int",
                "min": 4,
                "max": 80,
                "default": 12,
            }
        ]

    def pointer_down(self, session, event) -> None:
        layer = session.active_layer()
        if layer is None or layer.kind == "redaction":
            session.show_message("Select a raster or text layer to mark.")
            return
        before = visual_state(layer)
        image = layer.image.copy()
        draw = ImageDraw.Draw(image)
        x = event.x - layer.x
        y = event.y - layer.y
        size = int(session.option("marker_size", 12))
        color = session.foreground
        draw.line([(x - size, y), (x + size, y)], fill=color, width=max(1, size // 6))
        draw.line([(x, y - size), (x, y + size)], fill=color, width=max(1, size // 6))
        layer.image = image
        layer.kind = "raster"
        layer.text = None
        new_state = visual_state(layer)
        apply_visual(layer, before)
        session.commit(MutateLayerCommand(layer, new_state, name="Marker"))


def register(api) -> None:
    api.add_command(
        command_id="example.vignette",
        label="Vignette",
        menu="Filter",
        callback=apply_vignette,
    )
    api.add_tool(MarkerTool())
