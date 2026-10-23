"""Editing session shared by the window, tools, and plugins.

The session owns the document, the undo history, the foreground color, and
tool options. It does not import Qt. The window listens for ``change`` and
``preview`` notifications and redraws.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable

import numpy as np
from PIL import Image

from pixedit.composite import composite, merge_pair
from pixedit.factory import blank_layer, empty_raster_layer, raster_layer, redaction_layer, text_layer
from pixedit.geom import (
    alpha_bbox,
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
    CompoundCommand,
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
    clamp_mask_amount,
    new_document,
    visual_state,
)
from pixedit.textutil import default_font_path, render_text

Listener = Callable[[str], None]

_COLOR_OPTION_KEYS = (
    "color_hue",
    "color_saturation",
    "color_brightness",
    "color_contrast",
    "color_highlight",
    "color_shade",
)


def default_options() -> dict[str, Any]:
    return {
        "brush_size": 24,
        "brush_hardness": 85,
        "brush_opacity": 100,
        "mask_shape": "rect",
        "mask_kind": "pixelate",
        "mask_amount": 12,
        "color_hue": 0,
        "color_saturation": 0,
        "color_brightness": 0,
        "color_contrast": 0,
        "color_highlight": 0,
        "color_shade": 0,
        "select_mode": "rect",
        "tolerance": 32,
        "shape_kind": "rect",
        "stroke_width": 4,
        "shape_filled": False,
        "marker_size": 12,
        "rulers": True,
        "text_font": default_font_path(),
        "text_font_family": "DejaVu Sans",
        "text_size": 48,
        "text_align": "left",
        "text_line_spacing": 4,
        "text_rotation": 0,
    }


class Session:
    def __init__(self, document: Document | None = None) -> None:
        self.document = document or new_document(1280, 800, "white")
        self.history = History()
        # Set before any layer_index assignment. The setter drops an in-progress
        # color preview when the active layer changes, and it reads this flag.
        self._color_before: dict[str, Any] | None = None
        self._color_layer: Layer | None = None
        self._ending_color = False
        self._layer_index = len(self.document.layers) - 1
        self.foreground: tuple[int, int, int, int] = (20, 24, 28, 255)
        self.options = default_options()
        self.path: str | None = None
        self.dirty = False
        self.view_zoom = 1.0
        self._listeners: list[Listener] = []
        self._message_handler: Callable[[str], None] | None = None
        self._text_prompter: Callable[[TextInfo | None], TextInfo | None] | None = None
        self._redact_before: RedactionInfo | None = None
        self._text_before: dict | None = None

    @property
    def layer_index(self) -> int:
        return self._layer_index

    @layer_index.setter
    def layer_index(self, value: int) -> None:
        """Select a layer. A color preview belongs to one layer, so leaving it restores that layer."""
        new_index = int(value)
        preview = self._color_layer
        self._layer_index = new_index
        if preview is None:
            return
        current = None
        if self.document.layers and 0 <= new_index < len(self.document.layers):
            current = self.document.layers[new_index]
        if current is not preview:
            self.cancel_color_adjust()

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
        # An unapplied color grade is not part of the next undo step. The
        # notify below reloads the sliders after the options are zeroed.
        self._discard_color_preview()
        self.history.push(command)
        self.dirty = True
        self.clamp_index()
        self.notify("change")

    def undo(self) -> None:
        self._discard_color_preview()
        if self.history.undo():
            self.dirty = True
            self.clamp_index()
            self.notify("change")

    def redo(self) -> None:
        self._discard_color_preview()
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
        self.move_layer(self.layer_index, self.layer_index + delta)

    def move_layer(self, old_index: int, new_index: int) -> None:
        """Move one layer from ``old_index`` to ``new_index``.

        Index 0 is the back of the stack (painted first). A higher index is
        in front. One undo step. The moved layer stays selected.
        """
        count = len(self.document.layers)
        if not (0 <= old_index < count and 0 <= new_index < count):
            return
        if old_index == new_index:
            return
        self.commit(ReorderLayerCommand(self.document, old_index, new_index))
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
            self.show_message("No selection. Draw one with Select (M), then Delete clears its pixels.")
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
            self.show_message("No selection. Draw one with Select (M). Esc or Ctrl+D deselects.")
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
            self.show_message("Draw a selection first. Select (M), then Ctrl+C copies it to a new layer.")
            return
        layer = self.active_layer()
        if layer is None:
            self.show_message("Choose a layer to copy from.")
            return
        if layer.kind == "redaction":
            self.show_message("A mask is not pixels. Copy the layer underneath it.")
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

    def place_layers(self, layers: list[Layer]) -> None:
        """Add image layers on top of the current document, centered.

        The document size and the layers already there stay put. Each placed
        bitmap is cropped to its opaque pixels, then centered. Transparent
        padding in a PNG does not become a full-canvas plate. Undo removes
        every layer this call added.
        """
        copies = [layer.clone() for layer in layers if layer.kind != "redaction"]
        if not copies:
            self.show_message("That file has no image layers to place.")
            return
        index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
        commands = []
        for offset, layer in enumerate(copies):
            tighten_raster_layer(layer)
            layer.x = int(round((self.document.width - layer.image.width) / 2))
            layer.y = int(round((self.document.height - layer.image.height) / 2))
            commands.append(InsertLayerCommand(self.document, layer, index + offset, name="Place image"))
        self.commit(CompoundCommand(commands, "Place image"))
        # The first placed frame is the visible one for a multi-frame file.
        self.layer_index = index
        count = len(copies)
        noun = "layer" if count == 1 else "layers"
        self.show_message(f"Placed {count} {noun}. The current document was kept.")

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

    def crop_to_selection(self) -> None:
        """Crop the canvas to the selection's opaque bounds.

        The Image menu and the Select panel share this path. An empty
        selection explains itself and leaves the canvas alone.
        """
        selection = self.document.selection
        if selection is None:
            self.show_message("Draw a selection first.")
            return
        box = alpha_bbox(selection)
        if box is None:
            self.show_message("The selection is empty.")
            return
        left, top, right, bottom = box
        self.crop_to(left, top, right - left, bottom - top)

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

    def flip_active_horizontal(self) -> None:
        self.transform_active("flip-h")

    def flip_active_vertical(self) -> None:
        self.transform_active("flip-v")

    def rotate_active_cw(self) -> None:
        self.transform_active("rotate", degrees=90)

    def rotate_active_ccw(self) -> None:
        self.transform_active("rotate", degrees=-90)

    def transform_active(self, kind: str, **kwargs: Any) -> None:
        self._discard_color_preview()
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
        self._discard_color_preview()
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

        The new layer is inserted for preview only. The mask tool removes it
        and re-adds it through ``add_layer`` on release so undo drops the whole
        region in one step.
        """
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is not None and layer.kind == "redaction":
            return layer, False
        mask = Image.new("L", (self.document.width, self.document.height), 0)
        info = self.current_mask_info()
        created = redaction_layer(
            "Mask",
            mask,
            info.kind,
            info.amount,
            color=info.color,
            seed=info.seed,
        )
        index = len(self.document.layers) if self.layer_index < 0 else self.layer_index + 1
        self.document.layers.insert(index, created)
        self.layer_index = index
        return created, True

    def current_mask_info(self, previous: RedactionInfo | None = None) -> RedactionInfo:
        """Effect settings from the Mask panel.

        A new region gets a fresh noise seed. Another stroke on the same layer
        keeps that seed, and keeps the stored color unless the effect is Solid,
        which takes the foreground swatch.
        """
        kind = str(self.option("mask_kind", "pixelate"))
        amount = clamp_mask_amount(kind, float(self.option("mask_amount", 12)))
        if kind == "solid":
            color = tuple(int(channel) for channel in self.foreground)
        elif previous is not None:
            color = tuple(int(channel) for channel in previous.color)
        else:
            color = (0, 0, 0, 255)
        seed = int(previous.seed) if previous is not None else random.randrange(1, 2**31)
        return RedactionInfo(kind, amount, color, seed)

    def set_redaction_settings(
        self,
        kind: str,
        amount: float,
        *,
        transient: bool,
        color: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Update the active mask layer while a slider is dragged, then commit."""
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is None or layer.kind != "redaction" or layer.redaction is None:
            self._redact_before = None
            return
        base = self._redact_before or layer.redaction
        chosen = tuple(int(channel) for channel in (color if color is not None else base.color))
        amount = clamp_mask_amount(str(kind), float(amount))
        updated = RedactionInfo(str(kind), amount, chosen, int(base.seed))
        if transient:
            if self._redact_before is None:
                self._redact_before = copy.deepcopy(layer.redaction)
            layer.redaction = updated
            self.preview()
            return
        old = self._redact_before or copy.deepcopy(layer.redaction)
        self._redact_before = None
        layer.redaction = old
        if old.kind == kind and abs(old.amount - amount) < 1e-6 and tuple(int(channel) for channel in old.color) == chosen:
            self.preview()
            return
        self.commit(SetRedactionSettings(layer, str(kind), amount, chosen))

    def recolor_solid_mask(self, *, transient: bool) -> None:
        """While the Mask tool is active, the foreground swatch is Solid's color."""
        layer = self.active_layer()
        if layer is None or layer.kind != "redaction" or layer.redaction is None:
            return
        if layer.redaction.kind != "solid":
            return
        self.set_redaction_settings(
            layer.redaction.kind,
            layer.redaction.amount,
            transient=transient,
            color=tuple(int(channel) for channel in self.foreground),
        )

    def add_text(self, info: TextInfo, x: int, y: int) -> None:
        if not info.text.strip():
            return
        layer = text_layer("Text", info, x, y)
        self.add_layer(layer, name="Add text")

    def text_info_from_options(self, text: str) -> TextInfo:
        """Words plus the Text panel's font, align, spacing, rotation, and palette color."""
        red, green, blue, alpha = (int(channel) for channel in self.foreground)
        align = str(self.option("text_align", "left") or "left")
        if align not in ("left", "center", "right", "justify"):
            align = "left"
        return TextInfo(
            text=text,
            font_path=self.option("text_font"),
            font_family=str(self.option("text_font_family") or "Default"),
            size=max(1, int(self.option("text_size", 48))),
            color=(red, green, blue, alpha),
            rotation=float(self.option("text_rotation", 0)),
            align=align,
            line_spacing=max(0, int(self.option("text_line_spacing", 4))),
        )

    def adopt_text_style(self, info: TextInfo) -> None:
        """Copy a text layer's style into the tool options and the palette.

        This does not render. Loading a layer into the panel must not restyle it.
        """
        color = tuple(int(channel) for channel in info.color)
        if len(color) == 3:
            color = (*color, 255)
        self.options["text_font"] = info.font_path
        self.options["text_font_family"] = info.font_family or "Default"
        self.options["text_size"] = max(1, int(info.size))
        self.options["text_align"] = info.align if info.align in ("left", "center", "right", "justify") else "left"
        self.options["text_line_spacing"] = max(0, int(info.line_spacing))
        self.options["text_rotation"] = float(info.rotation)
        self.foreground = color

    def text_edit_in_progress(self) -> bool:
        return self._text_before is not None

    def apply_text_style(self, *, transient: bool) -> None:
        """Restyle the active text layer from the panel and the foreground color.

        A drag (rotation, line spacing, or the palette) previews on every move
        and commits once on release. The preview is not an undo step. Releasing
        on the same style the drag started from leaves history alone.

        Flow:
        1. No active text layer: forget any preview and return.
        2. Build the style from the panel options plus the palette color,
           keeping the layer's words.
        3. While dragging, remember the pre-drag pixels on the first move,
           draw the preview, and stop.
        4. On release, put those pre-drag pixels back, then commit one edit
           if the style actually changed.
        """
        layer = self.active_layer()
        if layer is None or layer.kind != "text" or layer.text is None:
            self._text_before = None
            return
        info = self.text_info_from_options(layer.text.text)
        if transient:
            if self._text_before is None:
                self._text_before = visual_state(layer)
            if _text_same(layer.text, info):
                return
            self._write_text(layer, info)
            self.preview()
            return
        before = self._text_before
        self._text_before = None
        if before is not None:
            apply_visual(layer, before)
        if layer.text is None or _text_same(layer.text, info):
            if before is not None:
                self.preview()
            return
        self.update_text(layer, info)

    def _write_text(self, layer: Layer, info: TextInfo) -> None:
        image = render_text(info)
        # Keep the center stable when the render size changes (rotation, font size).
        cx = layer.x + layer.image.width / 2
        cy = layer.y + layer.image.height / 2
        layer.image = image
        layer.text = copy.deepcopy(info)
        layer.kind = "text"
        layer.x = int(round(cx - image.width / 2))
        layer.y = int(round(cy - image.height / 2))

    def update_text(self, layer: Layer, info: TextInfo) -> None:
        before = visual_state(layer)
        self._write_text(layer, info)
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name="Edit text"))

    def color_adjust_values(self) -> dict[str, float]:
        return {key: float(self.option(key, 0)) for key in _COLOR_OPTION_KEYS}

    def color_adjust_active(self) -> bool:
        return self._color_before is not None

    def _color_is_neutral(self, values: dict[str, float] | None = None) -> bool:
        data = self.color_adjust_values() if values is None else values
        return all(abs(value) < 1e-6 for value in data.values())

    def _zero_color_options(self) -> None:
        for key in _COLOR_OPTION_KEYS:
            self.options[key] = 0

    def _restore_color_snapshot(self) -> bool:
        before = self._color_before
        layer = self._color_layer
        self._color_before = None
        self._color_layer = None
        if before is None or layer is None:
            return False
        apply_visual(layer, before)
        return True

    def _discard_color_preview(self) -> None:
        """Put the snapshot back and zero the sliders. The caller notifies."""
        self._restore_color_snapshot()
        self._zero_color_options()

    def cancel_color_adjust(self) -> bool:
        """Drop an unapplied grade. True when a preview was actually showing.

        Escape uses that flag so a preview is cancelled first and a selection
        is cleared only when nothing was previewing. Switching tools and
        selecting another layer use the same restore.
        """
        if self._ending_color:
            return False
        self._ending_color = True
        try:
            neutral = self._color_is_neutral()
            restored = self._restore_color_snapshot()
            self._zero_color_options()
            if restored or not neutral:
                self.notify("change")
            return restored
        finally:
            self._ending_color = False

    def preview_color_adjust(self) -> None:
        """Redraw the active layer from the sliders without writing history.

        Flow:
        1. A mask layer is refused. The sliders return to zero.
        2. All sliders at zero put the snapshot back, if a drag had started.
        3. The first non-zero value snapshots the layer, including text.
        4. Later values recompute from that snapshot so the sliders combine
           into one grade. The selection, when there is one, limits the grade.
        """
        layer = self.active_layer()
        if layer is None or layer.kind == "redaction":
            self.cancel_color_adjust()
            self.show_message("Choose a raster or text layer for this filter.")
            return
        if self._color_layer is not None and layer is not self._color_layer:
            self.cancel_color_adjust()
            return
        values = self.color_adjust_values()
        if self._color_is_neutral(values):
            if self._color_before is not None and self._color_layer is layer:
                apply_visual(layer, self._color_before)
                self._color_before = None
                self._color_layer = None
                self.preview()
            return
        if self._color_before is None:
            self._color_before = visual_state(layer)
            self._color_layer = layer
        self._paint_color_preview(layer)
        self.preview()

    def _paint_color_preview(self, layer: Layer) -> None:
        from pixedit.adjust import apply_masked, color_grade

        before = self._color_before
        if before is None:
            return
        values = self.color_adjust_values()
        edited = color_grade(
            before["image"],
            hue=values["color_hue"],
            saturation=values["color_saturation"],
            brightness=values["color_brightness"],
            contrast=values["color_contrast"],
            highlight=values["color_highlight"],
            shade=values["color_shade"],
        )
        selection = selection_for_layer(self.document, layer)
        layer.image = apply_masked(before["image"], edited, selection)
        # Text is shown as pixels for the preview. Cancel restores ``before``,
        # which still has the text metadata. Apply keeps the pixels.
        layer.kind = "raster"
        layer.text = None

    def commit_color_adjust(self) -> None:
        """Keep the preview as one undo step and return the sliders to zero."""
        layer = self._color_layer
        before = self._color_before
        if layer is None or before is None or self._color_is_neutral():
            self.cancel_color_adjust()
            return
        self._paint_color_preview(layer)
        new_state = visual_state(layer)
        new_state["kind"] = "raster"
        new_state["text"] = None
        apply_visual(layer, before)
        self._color_before = None
        self._color_layer = None
        self._zero_color_options()
        self.commit(MutateLayerCommand(layer, new_state, name="Color"))

    def run_filter(self, name: str, function: Callable[[Image.Image], Image.Image]) -> None:
        """Apply a pixel function to the active raster or text layer, honoring the selection."""
        self._discard_color_preview()
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


def _text_same(left: TextInfo, right: TextInfo) -> bool:
    return (
        left.text == right.text
        and left.font_path == right.font_path
        and left.font_family == right.font_family
        and int(left.size) == int(right.size)
        and tuple(int(channel) for channel in left.color) == tuple(int(channel) for channel in right.color)
        and abs(float(left.rotation) - float(right.rotation)) < 1e-4
        and (left.align or "left") == (right.align or "left")
        and int(left.line_spacing) == int(right.line_spacing)
    )
