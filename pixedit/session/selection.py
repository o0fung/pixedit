"""The pixel selection: draw, clear, cut, copy, and feather.

Mixed into ``Session``. This is the marching outline on the canvas, not which
layer row is selected. That row selection lives in ``core``.

A selection is coverage, not a hard inside/outside. 255 includes a pixel, 0
excludes it, and a value between fades paint, fill, and filters. Feather
blurs that mask. The marching outline is drawn on the 50% edge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pixedit.factory import raster_layer
from pixedit.geom import (
    alpha_bbox,
    covers_document,
    place_image,
    selection_for_layer,
    tighten_raster_layer,
)
from pixedit.model import (
    CompoundCommand,
    InsertLayerCommand,
    Layer,
    MutateLayerCommand,
    SetSelectionCommand,
    apply_visual,
    slot_above_block,
    visual_state,
)
from pixedit.pixels import gaussian_image

# Feather asks for a pixel radius. Wider than this, a small selection blurs
# away to nothing, so the dialog and the session share the same cap.
FEATHER_MIN = 1
FEATHER_MAX = 250


def _coverage(local: Image.Image, shape: tuple[int, int]) -> np.ndarray | None:
    """Layer-local selection as 0..1 coverage, or None when it misses the bitmap."""
    strength = np.asarray(local.convert("L")).astype(np.float32) / 255.0
    if strength.shape != shape or not bool(np.any(strength > 0)):
        return None
    return strength


def _fade_coverage(arr: np.ndarray, strength: np.ndarray) -> None:
    """Drop coverage in place. Full strength clears color and alpha.

    A value between 0 and 1 keeps that fraction of the alpha and the color.
    Pixels the selection does not cover stay as they were, including any
    color stored in a pixel that was already transparent.
    """
    covered = strength > 0
    alpha = arr[..., 3].astype(np.float32)
    faded = np.clip(np.rint(np.where(covered, alpha * (1.0 - strength), alpha)), 0, 255).astype(np.uint8)
    removed = covered & (faded == 0)
    arr[..., 3] = faded
    arr[removed, :3] = 0


def _covered_crop(source: np.ndarray, strength: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    """Tight RGBA of ``source`` scaled by ``strength``, plus its local origin.

    None when the overlap has no opaque pixels. Alpha is multiplied by the
    coverage, so a feathered edge stays soft and a hard edge still drops
    every pixel outside the selection.
    """
    extracted = source.astype(np.float32)
    extracted[..., 3] *= strength
    extracted[extracted[..., 3] <= 0.0, :3] = 0
    pixels = np.clip(np.rint(extracted), 0, 255).astype(np.uint8)
    opaque = pixels[..., 3] > 0
    if not bool(opaque.any()):
        return None
    ys, xs = np.nonzero(opaque)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return pixels[y0:y1, x0:x1], x0, y0


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

        Text is rasterized in the same undo step. Coverage fades the alpha, so
        a feathered edge thins the pixels instead of cutting a hard hole.
        Full coverage still clears color and alpha. A layer that already
        covers the document keeps its size, so a hole stays a hole in the
        plate. Smaller layers are cropped back to whatever opaque pixels
        remain. Redaction masks are left alone.
        """
        if not self.require_page():
            return
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
        strength = self._layer_coverage(layer)
        if strength is None:
            return
        before = visual_state(layer)
        preserve = covers_document(
            layer.x, layer.y, layer.image.width, layer.image.height, self.document.width, self.document.height
        )
        arr = np.array(layer.image.convert("RGBA"))
        _fade_coverage(arr, strength)
        layer.image = Image.fromarray(arr, "RGBA")
        layer.kind = "raster"
        layer.text = None
        if not preserve:
            tighten_raster_layer(layer)
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name="Clear selection"))

    def select_all(self) -> None:
        if not self.require_page():
            return
        mask = Image.new("L", (self.document.width, self.document.height), 255)
        self.set_selection(mask, name="Select all")

    def deselect(self) -> None:
        if not self.require_page():
            return
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
        # bitmap changes. Copy leaves a text layer editable. Coverage scales
        # the copied alpha, so a feathered edge is copied soft.
        if not self.require_page():
            return
        if self.document.selection is None:
            self.show_message("Draw a selection first. Select (S), then Ctrl+C copies it to a new layer.")
            return
        layer = self.active_layer()
        if not self._can_copy_pixels(layer):
            return
        assert layer is not None
        cropped = self._cropped_coverage(layer)
        if cropped is None:
            return
        pixels, x0, y0 = cropped
        placed = raster_layer(f"{layer.name} copy", Image.fromarray(pixels, "RGBA"), layer.x + x0, layer.y + y0)
        index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
        index = slot_above_block(self.document.layers, index)
        name = "Cut selection" if clear else "Copy selection"
        commands: list[Any] = []
        if clear:
            before = visual_state(layer)
            cleared = np.array(layer.image.convert("RGBA"))
            strength = self._layer_coverage(layer)
            if strength is None:
                return
            _fade_coverage(cleared, strength)
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

    def image_for_clipboard(self) -> Image.Image | None:
        """PNG pixels for Edit → Copy to clipboard.

        A selection copies that coverage of the active layer, cropped to the
        opaque pixels. With no selection, the active layer's bitmap is copied
        as it is stored. The layer itself is not changed. None means the
        status bar already says why.
        """
        if not self.require_page():
            return None
        layer = self.active_layer()
        if not self._can_copy_pixels(layer):
            return None
        assert layer is not None
        if self.document.selection is not None:
            cropped = self._cropped_coverage(layer)
            if cropped is None:
                return None
            pixels, _x0, _y0 = cropped
            self.show_message("Copied the selection to the clipboard.")
            return Image.fromarray(pixels, "RGBA")
        image = layer.image.convert("RGBA")
        if alpha_bbox(image) is None:
            self.show_message("The active layer has no opaque pixels.")
            return None
        self.show_message("Copied the active layer to the clipboard.")
        return image.copy()

    def feather_selection(self, radius: float) -> None:
        """Blur the selection by ``radius`` pixels so edits fade at the edge.

        With no page open, stop before touching a mask. Otherwise:
        1. Clamp the radius to 1..250 px. The dialog uses the same range, and
           a larger request is brought back into it.
        2. Blur the page-sized mask. Paint, fill, and filters already treat
           the gray values as coverage.
        3. If that blur rounds away to nothing, leave the selection alone and
           say so. Applying it would make every later edit a no-op.
        4. If the mask did not change, do not add an undo step.
        """
        if not self.require_page():
            return
        if self.document.selection is None:
            self.show_message("No selection. Draw one with Select (S), then Feather softens its edge.")
            return
        radius = int(round(float(radius)))
        radius = min(FEATHER_MAX, max(FEATHER_MIN, radius))
        full = place_image(self.document.selection.convert("L"), 0, 0, self.document.width, self.document.height, 0)
        blurred = np.asarray(gaussian_image(full, radius).convert("L"))
        if int(blurred.max()) == 0:
            self.show_message("That feather clears the selection. Try a smaller radius.")
            return
        if np.array_equal(np.asarray(full), blurred):
            self.show_message(f"Feather of {radius} px did not change the selection.")
            return
        self.set_selection(Image.fromarray(np.ascontiguousarray(blurred), "L"), name="Feather")
        self.show_message(f"Feathered the selection by {radius} px. The outline is the 50% edge.")

    def _can_copy_pixels(self, layer: Layer | None) -> bool:
        if layer is None:
            self.show_message("Choose a layer to copy from.")
            return False
        if layer.kind == "redaction":
            self.show_message("A mask is not pixels. Copy the layer it belongs to.")
            return False
        return True

    def _layer_coverage(self, layer: Layer) -> np.ndarray | None:
        """Coverage for ``layer``, or None after explaining a miss."""
        local = selection_for_layer(self.document, layer)
        if local is None:
            self.show_message("The selection does not overlap the active layer.")
            return None
        strength = _coverage(local, layer.image.size)
        if strength is None:
            self.show_message("The selection does not overlap the active layer.")
            return None
        return strength

    def _cropped_coverage(self, layer: Layer) -> tuple[np.ndarray, int, int] | None:
        """Selected pixels of ``layer``, cropped to what is still opaque."""
        strength = self._layer_coverage(layer)
        if strength is None:
            return None
        source = np.array(layer.image.convert("RGBA"))
        cropped = _covered_crop(source, strength)
        if cropped is None:
            self.show_message("The selection has no opaque pixels on the active layer.")
            return None
        return cropped

    def invert_selection(self) -> None:
        if not self.require_page():
            return
        if self.document.selection is None:
            self.set_selection(Image.new("L", (self.document.width, self.document.height), 0), name="Invert selection")
            return
        full = place_image(self.document.selection, 0, 0, self.document.width, self.document.height, 0)
        inverted = Image.eval(full, lambda value: 255 - value)
        self.set_selection(inverted, name="Invert selection")
