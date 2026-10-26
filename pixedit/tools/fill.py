"""Fill tool. Flood-fills the active layer from the pixel you click."""

from __future__ import annotations

import math

from pixedit.geom import place_image
from pixedit.paint import fill_document
from pixedit.tools.base import (
    PointerEvent,
    Tool,
)


def _canvas_selected(document, x: int, y: int) -> bool:
    """Whether document pixel ``(x, y)`` is inside the current selection."""
    selection = document.selection
    if selection is None:
        return True
    if selection.size != (document.width, document.height):
        selection = place_image(selection, 0, 0, document.width, document.height, 0)
    if not (0 <= x < selection.width and 0 <= y < selection.height):
        return False
    return int(selection.getpixel((x, y))) > 0


class FillTool(Tool):
    id = "fill"
    label = "Fill"
    shortcut = "G"
    cursor = "bucket"
    hint = (
        "Click to flood-fill connected pixels within tolerance, including empty space "
        "outside the layer's pixels. The fill stops at a different pixel or the canvas edge. "
        "A selection limits the fill."
    )

    def option_schema(self):
        return [
            {"key": "tolerance", "label": "Tolerance", "type": "int", "min": 0, "max": 255, "default": 32},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        # A click anywhere on the canvas is a seed, including empty space the
        # layer bitmap does not cover. Outside the canvas or the selection:
        # explain and do not commit. A click outside the selection must not
        # fill a connected region that merely overlaps the selection elsewhere.
        doc = session.document
        px = int(math.floor(event.x))
        py = int(math.floor(event.y))
        if not (0 <= px < doc.width and 0 <= py < doc.height):
            session.show_message("Click is outside the canvas.")
            return
        layer, before, created = session.begin_raster_edit(paint_redaction=False)
        if doc.selection is not None and not _canvas_selected(doc, px, py):
            session.cancel_raster_edit(layer, before, created)
            session.show_message("Click is outside the selection. Fill only changes pixels inside it.")
            return
        # fill_document also pulls the soft inner edge of a brush stroke into
        # the fill, so a closed outline does not keep a rim of the old background.
        layer.image, layer.x, layer.y = fill_document(
            layer.image,
            layer.x,
            layer.y,
            doc.width,
            doc.height,
            px,
            py,
            int(session.option("tolerance", 32)),
            session.foreground,
            doc.selection,
        )
        session.finish_raster_edit(layer, before, created, "Fill")
