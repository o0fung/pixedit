"""Brush, fill, and shape gestures, plus mask layers.

Mixed into ``Session``. A stroke previews on the layer, then ``finish_raster_edit``
turns it into one undo step. Cancelling puts the layer back.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from PIL import Image

from pixedit.factory import empty_raster_layer, redaction_layer
from pixedit.geom import covers_document, tighten_raster_layer
from pixedit.model import (
    Layer,
    MutateLayerCommand,
    RedactionInfo,
    SetRedactionSettings,
    apply_visual,
    clamp_mask_amount,
    contiguous_mask_span,
    slot_above_block,
    visual_state,
)


class RasterMixin:
    """Start and finish a paint stroke, and edit mask settings."""

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
            stack = self._stack()
            created = empty_raster_layer(f"Layer {len(stack) + 1}")
            index = len(stack) if self.layer_index < 0 else self.layer_index + 1
            index = slot_above_block(stack, index)
            stack.insert(index, created)
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
            stack = self._stack()
            try:
                index = stack.index(layer)
            except ValueError:
                index = len(stack)
            else:
                stack.pop(index)
            self.add_layer(layer, index=index, name=name)
            return
        if before is None:
            return
        new_state = visual_state(layer)
        apply_visual(layer, before)
        self.commit(MutateLayerCommand(layer, new_state, name=name))

    def cancel_raster_edit(self, layer: Layer, before: dict[str, Any] | None, created: bool) -> None:
        if created:
            self._replace_stack([item for item in self._stack() if item is not layer])
            self.clamp_index()
            self.preview()
            return
        if before is not None:
            apply_visual(layer, before)
            self.preview()

    def ensure_redaction_layer(self) -> tuple[Layer, bool]:
        """Return the active mask, or create one for the active picture.

        A second stroke on a mask adds to that mask and keeps its picture.
        A stroke on a picture or text layer creates a mask for that layer,
        placed above any masks it already has. The new layer is inserted for
        preview only. The mask tool removes it and re-adds it through
        ``add_layer`` on release so undo drops the whole region in one step.
        """
        self._discard_color_preview()
        layer = self.active_layer()
        if layer is not None and layer.kind == "redaction":
            return layer, False
        target = layer if layer is not None and layer.kind in ("raster", "text") else None
        stack = self._stack()
        if target is None:
            for candidate in reversed(stack):
                if candidate.kind in ("raster", "text"):
                    target = candidate
                    break
        mask = Image.new("L", (self.document.width, self.document.height), 0)
        info = self.current_mask_info()
        created = redaction_layer(
            "Mask",
            mask,
            info.kind,
            info.amount,
            color=info.color,
            seed=info.seed,
            target_id=None if target is None else target.id,
        )
        if target is None:
            index = len(stack)
            self.show_message("Add a picture layer before drawing a mask.")
        else:
            span = contiguous_mask_span(stack, target)
            index = len(stack) if span is None else span[1] + 1
        stack.insert(index, created)
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
