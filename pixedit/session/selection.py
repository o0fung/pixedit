"""The pixel selection: draw, clear, cut, and copy.

Mixed into ``Session``. This is the marching outline on the canvas, not which
layer row is selected. That row selection lives in ``core``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pixedit.factory import raster_layer
from pixedit.geom import covers_document, place_image, selection_for_layer, tighten_raster_layer
from pixedit.model import (
    CompoundCommand,
    InsertLayerCommand,
    MutateLayerCommand,
    SetSelectionCommand,
    apply_visual,
    slot_above_block,
    visual_state,
)


class SelectionMixin:
    """Select pixels, clear them, or move them onto a new layer."""

    def set_selection(self, selection: Image.Image | None, name: str = "Selection") -> None:
        introducing = self.document.selection is None and selection is not None
        self.commit(SetSelectionCommand(self.document, selection, name=name))
        if introducing:
            self.show_message(
                "Selection clips brush, fill, shapes, and filters. "
                "Ctrl+X cuts it to a new layer · Ctrl+C copies it · Delete clears pixels · "
                "Ctrl+D or Esc deselects."
            )

    def clear_selection_contents(self) -> None:
        """Erase pixels of the active raster or text layer inside the selection.

        Text is rasterized in the same undo step. A layer that already covers
        the document keeps its size, so a hole stays a hole in the plate.
        Smaller layers are cropped back to whatever opaque pixels remain.
        Redaction masks are left alone.
        """
        if self.document.selection is None:
            self.show_message("No selection. Draw one with Select (S), then Delete clears its pixels.")
            return
        layer = self.active_layer()
        if layer is None:
            self.show_message("No layer to clear.")
            return
        if layer.kind == "redaction":
            self.show_message("Delete clears pixels on a raster layer. Use the eraser to lift a mask.")
            return
        before = visual_state(layer)
        local = selection_for_layer(self.document, layer)
        if local is None:
            self.show_message("The selection does not overlap the active layer.")
            return
        selected = np.asarray(local) > 0
        if not bool(selected.any()):
            self.show_message("The selection does not overlap the active layer.")
            return
        preserve = covers_document(
            layer.x, layer.y, layer.image.width, layer.image.height, self.document.width, self.document.height
        )
        arr = np.array(layer.image.convert("RGBA"))
        # Zero RGB as well as alpha so a later tight crop and any sampling
        # of the cleared pixels do not resurrect the old color.
        arr[selected, :3] = 0
        arr[selected, 3] = 0
        layer.image = Image.fromarray(arr, "RGBA")
        layer.kind = "raster"
        layer.text = None
        if not preserve:
            tighten_raster_layer(layer)
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name="Clear selection"))

    def select_all(self) -> None:
        mask = Image.new("L", (self.document.width, self.document.height), 255)
        self.set_selection(mask, name="Select all")

    def deselect(self) -> None:
        if self.document.selection is None:
            self.show_message("No selection. Draw one with Select (S). Esc or Ctrl+D deselects.")
            return
        self.set_selection(None, name="Deselect")
        self.show_message("Deselected.")

    def copy_selection_to_layer(self) -> None:
        """Copy opaque selected pixels onto a new layer above the active one."""
        self._selection_to_new_layer(clear=False)

    def cut_selection_to_layer(self) -> None:
        """Move opaque selected pixels onto a new layer and clear them on the source."""
        self._selection_to_new_layer(clear=True)

    def _selection_to_new_layer(self, *, clear: bool) -> None:
        # One undo step either duplicates the selected pixels or moves them.
        # Cut rasterizes text, the same way Delete does, because the source
        # bitmap changes. Copy leaves a text layer editable.
        if self.document.selection is None:
            self.show_message("Draw a selection first. Select (S), then Ctrl+C copies it to a new layer.")
            return
        layer = self.active_layer()
        if layer is None:
            self.show_message("Choose a layer to copy from.")
            return
        if layer.kind == "redaction":
            self.show_message("A mask is not pixels. Copy the layer it belongs to.")
            return
        local = selection_for_layer(self.document, layer)
        if local is None:
            self.show_message("The selection does not overlap the active layer.")
            return
        mask = np.asarray(local) > 0
        if not bool(mask.any()):
            self.show_message("The selection does not overlap the active layer.")
            return
        source = np.array(layer.image.convert("RGBA"))
        opaque = mask & (source[..., 3] > 0)
        ys, xs = np.nonzero(opaque)
        if len(xs) == 0:
            self.show_message("The selection has no opaque pixels on the active layer.")
            return
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        extracted = source.copy()
        extracted[~mask] = 0
        placed = raster_layer(
            f"{layer.name} copy",
            Image.fromarray(extracted[y0:y1, x0:x1], "RGBA"),
            layer.x + x0,
            layer.y + y0,
        )
        index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
        index = slot_above_block(self.document.layers, index)
        name = "Cut selection" if clear else "Copy selection"
        commands: list[Any] = []
        if clear:
            before = visual_state(layer)
            cleared = source.copy()
            cleared[mask, :3] = 0
            cleared[mask, 3] = 0
            scratch = layer.clone()
            scratch.image = Image.fromarray(cleared, "RGBA")
            scratch.kind = "raster"
            scratch.text = None
            full = covers_document(
                layer.x,
                layer.y,
                layer.image.width,
                layer.image.height,
                self.document.width,
                self.document.height,
            )
            if not full:
                tighten_raster_layer(scratch)
            commands.append(MutateLayerCommand(layer, visual_state(scratch), old_state=before, name=name))
        commands.append(InsertLayerCommand(self.document, placed, index, name=name))
        self.commit(CompoundCommand(commands, name))
        self.layer_index = index
        verb = "Cut" if clear else "Copied"
        self.show_message(f"{verb} the selection to a new layer. Esc or Ctrl+D deselects.")

    def invert_selection(self) -> None:
        if self.document.selection is None:
            self.set_selection(Image.new("L", (self.document.width, self.document.height), 0), name="Invert selection")
            return
        full = place_image(self.document.selection, 0, 0, self.document.width, self.document.height, 0)
        inverted = Image.eval(full, lambda value: 255 - value)
        self.set_selection(inverted, name="Invert selection")
