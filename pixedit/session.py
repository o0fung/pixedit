"""Editing session shared by the window, tools, and plugins.

The session owns the document, the undo history, the foreground color, and
tool options. It does not import Qt. The window listens for ``change`` and
``preview`` notifications and redraws.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
from PIL import Image

from pixedit.composite import composite, merge_pair
from pixedit.factory import blank_layer, empty_raster_layer, redaction_layer, text_layer
from pixedit.geom import (
    canvas_flip,
    canvas_rotate_ccw,
    canvas_rotate_cw,
    covers_document,
    crop_document,
    flip_visual,
    rotate_visual,
    scale_visual,
    selection_for_layer,
    tighten_raster_layer,
    trim_rect,
)
from pixedit.model import (
    DeleteLayerCommand,
    Document,
    DuplicateLayerCommand,
    History,
    InsertLayerCommand,
    Layer,
    MutateLayerCommand,
    RedactionInfo,
    ReorderLayerCommand,
    ReplaceDocumentContent,
    SetLayerAttrCommand,
    SetPositionCommand,
    SetRedactionSettings,
    SetSelectionCommand,
    TextInfo,
    apply_visual,
    new_document,
    visual_state,
)
from pixedit.textutil import render_text

Listener = Callable[[str], None]


def default_options() -> dict[str, Any]:
    return {
        "brush_size": 24,
        "brush_hardness": 85,
        "brush_opacity": 100,
        "redact_shape": "rect",
        "redact_kind": "pixelate",
        "redact_amount": 12,
        "select_mode": "rect",
        "tolerance": 32,
        "shape_kind": "rect",
        "stroke_width": 4,
        "shape_filled": False,
        "marker_size": 12,
        "rulers": True,
    }


class Session:
    def __init__(self, document: Document | None = None) -> None:
        self.document = document or new_document(1280, 800, "white")
        self.history = History()
        self.layer_index = len(self.document.layers) - 1
        self.foreground: tuple[int, int, int, int] = (20, 24, 28, 255)
        self.options = default_options()
        self.path: str | None = None
        self.dirty = False
        self.view_zoom = 1.0
        self._listeners: list[Listener] = []
        self._message_handler: Callable[[str], None] | None = None
        self._text_prompter: Callable[[TextInfo | None], TextInfo | None] | None = None
        self._redact_before: RedactionInfo | None = None

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def set_message_handler(self, handler: Callable[[str], None] | None) -> None:
        self._message_handler = handler

    def set_text_prompter(self, prompter: Callable[[TextInfo | None], TextInfo | None] | None) -> None:
        self._text_prompter = prompter

    def notify(self, kind: str = "change") -> None:
        for listener in list(self._listeners):
            listener(kind)

    def preview(self) -> None:
        """Redraw after a live edit that is not an undo step yet."""
        self.notify("preview")

    def show_message(self, text: str) -> None:
        if self._message_handler is not None:
            self._message_handler(text)

    def prompt_text(self, initial: TextInfo | None) -> TextInfo | None:
        if self._text_prompter is None:
            return None
        return self._text_prompter(initial)

    def commit(self, command: Any) -> None:
        self.history.push(command)
        self.dirty = True
        self.clamp_index()
        self.notify("change")

    def undo(self) -> None:
        if self.history.undo():
            self.dirty = True
            self.clamp_index()
            self.notify("change")

    def redo(self) -> None:
        if self.history.redo():
            self.dirty = True
            self.clamp_index()
            self.notify("change")

    def mark_clean(self) -> None:
        self.dirty = False
        self.notify("change")

    def clamp_index(self) -> None:
        if not self.document.layers:
            self.layer_index = -1
        else:
            self.layer_index = max(0, min(self.layer_index, len(self.document.layers) - 1))

    def active_layer(self) -> Layer | None:
        self.clamp_index()
        if self.layer_index < 0:
            return None
        return self.document.layers[self.layer_index]

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def replace_document(self, document: Document, path: str | None = None) -> None:
        self.document = document
        self.history = History()
        self.layer_index = len(document.layers) - 1
        self.path = path
        self.dirty = False
        self.notify("change")

    def new_canvas(self, width: int, height: int, background: str) -> None:
        self.replace_document(new_document(width, height, background), None)

    def add_layer(self, layer: Layer, index: int | None = None, name: str = "Add layer") -> None:
        if index is None:
            index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
        self.commit(InsertLayerCommand(self.document, layer, index, name=name))
        self.layer_index = max(0, min(index, len(self.document.layers) - 1))
        self.notify("change")

    def add_blank_layer(self) -> None:
        """Add an empty content-sized layer. The first stroke grows its bitmap."""
        number = len(self.document.layers) + 1
        layer = empty_raster_layer(f"Layer {number}")
        self.add_layer(layer, name="New layer")

    def duplicate_active(self) -> None:
        if self.layer_index < 0:
            return
        self.commit(DuplicateLayerCommand(self.document, self.layer_index))
        self.layer_index += 1

    def delete_active(self) -> None:
        if self.layer_index < 0 or not self.document.layers:
            return
        self.commit(DeleteLayerCommand(self.document, self.layer_index))
        self.clamp_index()

    def rename_active(self, name: str) -> None:
        layer = self.active_layer()
        if layer is None:
            return
        cleaned = name.strip() or layer.name
        if cleaned == layer.name:
            return
        self.commit(SetLayerAttrCommand(layer, "name", cleaned, name="Rename layer"))

    def move_active(self, delta: int) -> None:
        """``delta`` +1 moves the active layer up the stack (toward the front)."""
        if self.layer_index < 0:
            return
        new_index = self.layer_index + delta
        if not (0 <= new_index < len(self.document.layers)):
            return
        self.commit(ReorderLayerCommand(self.document, self.layer_index, new_index))
        self.layer_index = new_index

    def merge_down_active(self) -> None:
        if self.layer_index <= 0:
            self.show_message("Nothing below this layer to merge into.")
            return
        upper_index = self.layer_index
        lower = self.document.layers[upper_index - 1]
        upper = self.document.layers[upper_index]
        merged = merge_pair(lower, upper)
        merged.name = lower.name
        # Keep the lower layer's identity so a later undo restores both objects
        # that were in the stack, and the merged result is a third object.
        layers = list(self.document.layers)
        layers[upper_index - 1] = merged
        del layers[upper_index]
        self.commit(
            ReplaceDocumentContent(
                self.document,
                layers,
                self.document.width,
                self.document.height,
                self.document.selection,
                name="Merge down",
            )
        )
        self.layer_index = upper_index - 1

    def flatten(self) -> None:
        image = composite(self.document)
        layer = blank_layer("Flattened", self.document.width, self.document.height)
        layer.image = image
        self.commit(
            ReplaceDocumentContent(
                self.document,
                [layer],
                self.document.width,
                self.document.height,
                None,
                name="Flatten",
            )
        )
        self.layer_index = 0

    def set_visibility(self, layer: Layer, visible: bool) -> None:
        if layer.visible == visible:
            return
        self.commit(SetLayerAttrCommand(layer, "visible", visible, name="Toggle layer"))

    def set_opacity(self, layer: Layer, opacity: float) -> None:
        opacity = min(1.0, max(0.0, float(opacity)))
        if abs(layer.opacity - opacity) < 1e-6:
            return
        self.commit(SetLayerAttrCommand(layer, "opacity", opacity, name="Layer opacity"))

    def set_blend(self, layer: Layer, mode: str) -> None:
        if layer.blend_mode == mode:
            return
        self.commit(SetLayerAttrCommand(layer, "blend_mode", mode, name="Blend mode"))

    def set_selection(self, selection: Image.Image | None, name: str = "Selection") -> None:
        introducing = self.document.selection is None and selection is not None
        self.commit(SetSelectionCommand(self.document, selection, name=name))
        if introducing:
            self.show_message(
                "Selection clips brush, fill, shapes, and filters. "
                "Delete clears pixels · Ctrl+D deselects · Image → Crop to selection."
            )

    def clear_selection_contents(self) -> None:
        """Erase pixels of the active raster or text layer inside the selection.

        Text is rasterized in the same undo step. A layer that already covers
        the document keeps its size, so a hole stays a hole in the plate.
        Smaller layers are cropped back to whatever opaque pixels remain.
        Redaction masks are left alone.
        """
        if self.document.selection is None:
            self.show_message("No selection. Draw one with Select (M), then Delete clears its pixels.")
            return
        layer = self.active_layer()
        if layer is None:
            self.show_message("No layer to clear.")
            return
        if layer.kind == "redaction":
            self.show_message("Delete clears pixels on a raster layer. Use the eraser to lift a redaction mask.")
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
            return
        self.set_selection(None, name="Deselect")

    def invert_selection(self) -> None:
        if self.document.selection is None:
            self.set_selection(Image.new("L", (self.document.width, self.document.height), 0), name="Invert selection")
            return
        from pixedit.geom import place_image

        full = place_image(self.document.selection, 0, 0, self.document.width, self.document.height, 0)
        inverted = Image.eval(full, lambda value: 255 - value)
        self.set_selection(inverted, name="Invert selection")

    def crop_to(self, left: int, top: int, width: int, height: int) -> None:
        if width < 1 or height < 1:
            return
        layers, selection = crop_document(self.document, left, top, width, height)
        self.commit(ReplaceDocumentContent(self.document, layers, width, height, selection, name="Crop"))

    def trim(self) -> None:
        rect = trim_rect(self.document)
        if rect is None:
            self.show_message("Nothing opaque to trim to.")
            return
        left, top, width, height = rect
        if left == 0 and top == 0 and width == self.document.width and height == self.document.height:
            self.show_message("Canvas is already tight.")
            return
        self.crop_to(left, top, width, height)

    def transform_canvas(self, kind: str) -> None:
        if kind == "rotate-cw":
            layers, width, height, selection = canvas_rotate_cw(self.document)
            name = "Rotate canvas"
        elif kind == "rotate-ccw":
            layers, width, height, selection = canvas_rotate_ccw(self.document)
            name = "Rotate canvas"
        elif kind == "flip-h":
            layers, width, height, selection = canvas_flip(self.document, True)
            name = "Flip canvas"
        elif kind == "flip-v":
            layers, width, height, selection = canvas_flip(self.document, False)
            name = "Flip canvas"
        else:
            return
        self.commit(ReplaceDocumentContent(self.document, layers, width, height, selection, name=name))

    def transform_active(self, kind: str, **kwargs: Any) -> None:
        layer = self.active_layer()
        if layer is None:
            return
        state = visual_state(layer)
        if kind == "scale":
            new_state = scale_visual(state, int(kwargs["width"]), int(kwargs["height"]))
            name = "Scale"
        elif kind == "rotate":
            new_state = rotate_visual(state, float(kwargs["degrees"]))
            name = "Rotate"
        elif kind == "flip-h":
            new_state = flip_visual(state, True)
            name = "Flip horizontal"
        elif kind == "flip-v":
            new_state = flip_visual(state, False)
            name = "Flip vertical"
        else:
            return
        self.commit(MutateLayerCommand(layer, new_state, name=name))

    def move_layer_to(self, layer: Layer, x: int, y: int, old_x: int, old_y: int) -> None:
        if layer.x == old_x and layer.y == old_y and (x != old_x or y != old_y):
            pass
        if x == old_x and y == old_y:
            return
        self.commit(SetPositionCommand(layer, x, y, old_x=old_x, old_y=old_y))

    def apply_layer_image(self, layer: Layer, new_state: dict[str, Any], name: str, before: dict[str, Any] | None = None) -> None:
        if before is not None:
            apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, old_state=before, name=name))

    def begin_raster_edit(self, *, paint_redaction: bool = False) -> tuple[Layer, dict[str, Any] | None, bool]:
        """Prepare a layer for a brush, fill, or shape.

        Returns ``(layer, before_state, created)``. When ``created`` is true the
        layer was inserted immediately for preview and is not yet an undo step;
        ``finish_raster_edit`` turns that into one command. Text layers are
        rasterized after ``before_state`` is captured, so undo restores the text.

        The bitmap is not expanded to the document. Callers grow it around the
        gesture, and ``finish_raster_edit`` crops transparent margins unless
        the layer already covered the canvas.
        """
        layer = self.active_layer()
        if layer is not None and layer.kind == "redaction" and paint_redaction:
            return layer, visual_state(layer), False
        if layer is None or layer.kind == "redaction":
            created = empty_raster_layer(f"Layer {len(self.document.layers) + 1}")
            index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
            self.document.layers.insert(index, created)
            self.layer_index = index
            return created, None, True
        before = visual_state(layer)
        if layer.kind == "text":
            layer.kind = "raster"
            layer.text = None
        return layer, before, False

    def _fit_content_unless_full_canvas(self, layer: Layer, before: dict[str, Any] | None) -> None:
        """Crop a layer to its opaque pixels unless it was already a full-canvas plate.

        Redaction masks stay document-sized. A background, import, or flatten
        covers the document in ``before`` and keeps that extent even if the
        gesture only touched a corner.
        """
        if layer.kind == "redaction":
            return
        if before is not None:
            image = before["image"]
            if covers_document(
                int(before["x"]),
                int(before["y"]),
                image.width,
                image.height,
                self.document.width,
                self.document.height,
            ):
                return
        tighten_raster_layer(layer)

    def finish_raster_edit(
        self,
        layer: Layer,
        before: dict[str, Any] | None,
        created: bool,
        name: str,
    ) -> None:
        self._fit_content_unless_full_canvas(layer, before)
        if created:
            try:
                index = self.document.layers.index(layer)
            except ValueError:
                index = len(self.document.layers)
            else:
                self.document.layers.pop(index)
            self.add_layer(layer, index=index, name=name)
            return
        if before is None:
            return
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name=name))

    def cancel_raster_edit(self, layer: Layer, before: dict[str, Any] | None, created: bool) -> None:
        if created:
            self.document.layers = [item for item in self.document.layers if item is not layer]
            self.clamp_index()
            self.preview()
            return
        if before is not None:
            apply_visual(layer, before)
            self.preview()

    def ensure_redaction_layer(self) -> tuple[Layer, bool]:
        """Return the active redaction layer, creating one above the selection if needed.

        The new layer is inserted for preview only. The redact tool removes it
        and re-adds it through ``add_layer`` on release so undo drops the whole
        region in one step.
        """
        layer = self.active_layer()
        if layer is not None and layer.kind == "redaction":
            return layer, False
        mask = Image.new("L", (self.document.width, self.document.height), 0)
        created = redaction_layer(
            "Redaction",
            mask,
            str(self.option("redact_kind", "pixelate")),
            float(self.option("redact_amount", 12)),
        )
        index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
        self.document.layers.insert(index, created)
        self.layer_index = index
        return created, True

    def set_redaction_settings(self, kind: str, amount: float, *, transient: bool) -> None:
        """Update the active redaction layer while a slider is dragged, then commit."""
        layer = self.active_layer()
        if layer is None or layer.kind != "redaction" or layer.redaction is None:
            self._redact_before = None
            return
        amount = float(amount)
        if transient:
            if self._redact_before is None:
                self._redact_before = copy.deepcopy(layer.redaction)
            layer.redaction = RedactionInfo(kind, amount)
            self.preview()
            return
        old = self._redact_before or copy.deepcopy(layer.redaction)
        self._redact_before = None
        layer.redaction = old
        if old.kind == kind and abs(old.amount - amount) < 1e-6:
            self.preview()
            return
        self.commit(SetRedactionSettings(layer, kind, amount))

    def add_text(self, info: TextInfo, x: int, y: int) -> None:
        if not info.text.strip():
            return
        layer = text_layer("Text", info, x, y)
        self.add_layer(layer, name="Add text")

    def update_text(self, layer: Layer, info: TextInfo) -> None:
        before = visual_state(layer)
        image = render_text(info)
        # Keep the center stable when the render size changes (rotation, font size).
        cx = layer.x + layer.image.width / 2
        cy = layer.y + layer.image.height / 2
        layer.image = image
        layer.text = info
        layer.kind = "text"
        layer.x = int(round(cx - image.width / 2))
        layer.y = int(round(cy - image.height / 2))
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name="Edit text"))

    def run_filter(self, name: str, function: Callable[[Image.Image], Image.Image]) -> None:
        """Apply a pixel function to the active raster or text layer, honoring the selection."""
        layer = self.active_layer()
        if layer is None or layer.kind == "redaction":
            self.show_message("Choose a raster or text layer for this filter.")
            return
        before = visual_state(layer)
        from pixedit.adjust import apply_masked
        from pixedit.geom import selection_for_layer

        edited = function(layer.image)
        selection = selection_for_layer(self.document, layer)
        edited = apply_masked(layer.image, edited, selection)
        layer.image = edited
        layer.kind = "raster"
        layer.text = None
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name=name))
