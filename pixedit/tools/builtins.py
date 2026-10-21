"""Built-in canvas tools.

Tools mutate the session only through its command helpers. A gesture that
previews (brush, transform, redaction) snapshots the layer first and commits
a single undo step on release. Escape calls ``pointer_cancel``.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from PIL import Image

from pixedit.composite import composite
from pixedit.geom import (
    clamp_crop,
    combine_selection,
    hit_opaque,
    mask_ellipse,
    mask_freehand,
    mask_rect,
    place_image,
    rotate_visual,
    selection_for_layer,
    size_from_anchor,
)
from pixedit.model import Layer, RedactionInfo, TextInfo, apply_visual, visual_state
from pixedit.paint import (
    connected_mask,
    dab,
    dab_mask,
    draw_shape,
    fill_where,
    interpolate,
    sample_color,
)
from pixedit.textutil import default_font_path, render_text
from pixedit.tools.base import Overlay, PointerEvent, Tool

_FIXED = {"nw": "se", "ne": "sw", "sw": "ne", "se": "nw"}


def _brush_params(session) -> tuple[float, float, float]:
    # Size is the brush diameter in image pixels, matching the usual editor convention.
    radius = max(0.5, float(session.option("brush_size", 24)) / 2.0)
    hardness = float(session.option("brush_hardness", 85)) / 100.0
    opacity = float(session.option("brush_opacity", 100)) / 100.0
    return radius, hardness, opacity


def _local_selection(session, origin: tuple[int, int], shape: tuple[int, int]) -> np.ndarray | None:
    """Canvas selection cropped to a bitmap whose top-left is ``origin``."""
    if session.document.selection is None:
        return None
    height, width = shape
    sel = np.zeros((height, width), dtype=np.uint8)
    full = session.document.selection
    if full.size != (session.document.width, session.document.height):
        full = place_image(full, 0, 0, session.document.width, session.document.height, 0)
    ox, oy = origin
    x0 = max(0, ox)
    y0 = max(0, oy)
    x1 = min(session.document.width, ox + width)
    y1 = min(session.document.height, oy + height)
    if x1 <= x0 or y1 <= y0:
        return sel
    crop = np.asarray(full.crop((x0, y0, x1, y1)))
    sel[y0 - oy : y0 - oy + crop.shape[0], x0 - ox : x0 - ox + crop.shape[1]] = crop
    return sel


class MoveTool(Tool):
    id = "move"
    label = "Move"
    shortcut = "V"
    cursor = "size"
    hint = "Drag a layer. Corners scale, the handle above rotates. Shift locks aspect."

    def __init__(self) -> None:
        self._mode: str | None = None
        self._index = -1
        self._start_mouse = (0.0, 0.0)
        self._origin = (0, 0)
        self._start: dict | None = None
        self._anchor_name = ""
        self._anchor = (0.0, 0.0)
        self._start_angle = 0.0

    def pointer_down(self, session, event: PointerEvent) -> None:
        layer = session.active_layer()
        if layer is not None:
            handle = self._hit_handle(layer, event)
            if handle is not None:
                self._begin(session, session.layer_index, handle, event)
                return
        hit = hit_opaque(session.document, event.x, event.y)
        if hit is None:
            self._mode = None
            return
        session.layer_index = hit
        session.notify("change")
        self._begin(session, hit, "move", event)

    def _begin(self, session, index: int, mode: str, event: PointerEvent) -> None:
        layer = session.document.layers[index]
        self._mode = mode
        self._index = index
        self._start_mouse = (event.x, event.y)
        self._origin = (layer.x, layer.y)
        self._start = visual_state(layer)
        if mode in _FIXED:
            fixed = _FIXED[mode]
            handles = self._handle_points(layer, event.zoom)
            self._anchor_name = fixed
            ax, ay = handles[fixed]
            self._anchor = (ax, ay)
        elif mode == "rot":
            x, y, w, h = layer.bounds()
            self._start_angle = math.degrees(math.atan2(event.y - (y + h / 2), event.x - (x + w / 2)))

    def pointer_move(self, session, event: PointerEvent) -> None:
        if self._mode is None or self._start is None or self._index < 0:
            return
        layer = session.document.layers[self._index]
        if self._mode == "move":
            dx = event.x - self._start_mouse[0]
            dy = event.y - self._start_mouse[1]
            layer.x = int(round(self._origin[0] + dx))
            layer.y = int(round(self._origin[1] + dy))
        elif self._mode in _FIXED:
            self._scale_to(layer, event)
        elif self._mode == "rot":
            self._rotate_to(layer, event)
        session.preview()

    def _scale_to(self, layer: Layer, event: PointerEvent) -> None:
        assert self._start is not None
        start_image = self._start["mask"] if self._start["kind"] == "redaction" and self._start["mask"] is not None else self._start["image"]
        aspect = start_image.width / max(1, start_image.height)
        left, top, width, height = size_from_anchor(
            self._anchor,
            (event.x, event.y),
            lock_aspect=event.shift,
            aspect=aspect,
        )
        new_w = max(1, int(round(width)))
        new_h = max(1, int(round(height)))
        state = copy.deepcopy({k: v for k, v in self._start.items() if k not in {"image", "mask"}})
        state["image"] = self._start["image"].copy()
        state["mask"] = None if self._start["mask"] is None else self._start["mask"].copy()
        state["text"] = copy.deepcopy(self._start["text"])
        state["redaction"] = copy.deepcopy(self._start["redaction"])
        if self._start["kind"] == "text" and self._start["text"] is not None:
            info = copy.deepcopy(self._start["text"])
            ratio = new_w / max(1, self._start["image"].width)
            info.size = max(1, int(round(info.size * ratio)))
            image = render_text(info)
            pos = _anchor_top_left(self._anchor_name, self._anchor, image.width, image.height)
            state["text"] = info
            state["image"] = image
            state["kind"] = "text"
            state["x"], state["y"] = int(round(pos[0])), int(round(pos[1]))
        elif self._start["kind"] == "redaction" and self._start["mask"] is not None:
            state["mask"] = self._start["mask"].resize((new_w, new_h), Image.Resampling.NEAREST)
            pos = _anchor_top_left(self._anchor_name, self._anchor, new_w, new_h)
            state["x"], state["y"] = int(round(pos[0])), int(round(pos[1]))
        else:
            state["image"] = self._start["image"].resize((new_w, new_h), Image.Resampling.LANCZOS)
            state["kind"] = "raster"
            state["text"] = None
            pos = _anchor_top_left(self._anchor_name, self._anchor, new_w, new_h)
            state["x"], state["y"] = int(round(pos[0])), int(round(pos[1]))
        apply_visual(layer, state)

    def _rotate_to(self, layer: Layer, event: PointerEvent) -> None:
        assert self._start is not None
        x, y = self._start["x"], self._start["y"]
        image = self._start["mask"] if self._start["kind"] == "redaction" and self._start["mask"] is not None else self._start["image"]
        cx = x + image.width / 2
        cy = y + image.height / 2
        angle = math.degrees(math.atan2(event.y - cy, event.x - cx))
        # Screen Y grows downward, so atan2 increases clockwise — the editor's
        # positive rotation direction. Wrap so the gesture doesn't jump at ±180.
        delta = (angle - self._start_angle + 180.0) % 360.0 - 180.0
        apply_visual(layer, rotate_visual(self._start, delta))

    def pointer_up(self, session, event: PointerEvent) -> None:
        if self._mode is None or self._start is None or self._index < 0:
            return
        self.pointer_move(session, event)
        layer = session.document.layers[self._index]
        moved = math.hypot(event.x - self._start_mouse[0], event.y - self._start_mouse[1])
        mode = self._mode
        start = self._start
        origin = self._origin
        self._mode = None
        self._start = None
        if moved < 0.5:
            apply_visual(layer, start)
            session.preview()
            return
        if mode == "move":
            session.move_layer_to(layer, layer.x, layer.y, origin[0], origin[1])
            return
        new_state = visual_state(layer)
        apply_visual(layer, start)
        session.apply_layer_image(layer, new_state, "Transform", before=start)

    def pointer_cancel(self, session) -> None:
        if self._start is not None and 0 <= self._index < len(session.document.layers):
            apply_visual(session.document.layers[self._index], self._start)
            session.preview()
        self._mode = None
        self._start = None

    def overlays(self, session) -> list[Overlay]:
        layer = session.active_layer()
        if layer is None:
            return []
        zoom = float(getattr(session, "view_zoom", 1.0) or 1.0)
        points = self._handle_points(layer, zoom)
        x, y, w, h = layer.bounds()
        handles = [(name, px, py) for name, (px, py) in points.items()]
        return [Overlay(kind="handles", rect=(x, y, w, h), handles=handles)]

    def _handle_points(self, layer: Layer, zoom: float) -> dict[str, tuple[float, float]]:
        x, y, w, h = layer.bounds()
        lift = 28.0 / max(zoom, 0.05)
        return {
            "nw": (x, y),
            "ne": (x + w, y),
            "sw": (x, y + h),
            "se": (x + w, y + h),
            "rot": (x + w / 2.0, y - lift),
        }

    def _hit_handle(self, layer: Layer, event: PointerEvent) -> str | None:
        limit = 8.0 / max(event.zoom, 0.05)
        best = None
        best_dist = limit
        for name, (hx, hy) in self._handle_points(layer, event.zoom).items():
            dist = math.hypot(event.x - hx, event.y - hy)
            if dist <= best_dist:
                best = name
                best_dist = dist
        return best


def _anchor_top_left(anchor_name: str, anchor: tuple[float, float], width: float, height: float) -> tuple[float, float]:
    ax, ay = anchor
    if anchor_name == "nw":
        return ax, ay
    if anchor_name == "ne":
        return ax - width, ay
    if anchor_name == "sw":
        return ax, ay - height
    return ax - width, ay - height


class _PaintTool(Tool):
    """Shared stroke loop for the brush and the eraser."""

    erase = False
    cursor = "cross"

    def __init__(self) -> None:
        self._active = False
        self._last: tuple[float, float] | None = None
        self._layer: Layer | None = None
        self._before: dict | None = None
        self._created = False
        self._mask_mode = False
        self._array: np.ndarray | None = None
        self._selection: np.ndarray | None = None
        self._origin = (0, 0)

    def option_schema(self):
        return [
            {"key": "brush_size", "label": "Size", "type": "int", "min": 1, "max": 400, "default": 24},
            {"key": "brush_hardness", "label": "Hardness", "type": "int", "min": 0, "max": 100, "default": 85},
            {"key": "brush_opacity", "label": "Opacity", "type": "int", "min": 1, "max": 100, "default": 100},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        layer, before, created = session.begin_raster_edit(paint_redaction=True)
        self._layer = layer
        self._before = before
        self._created = created
        self._mask_mode = layer.kind == "redaction"
        self._origin = (layer.x, layer.y)
        if self._mask_mode:
            if layer.mask is None:
                layer.mask = Image.new("L", (session.document.width, session.document.height), 0)
                self._origin = (0, 0)
                layer.x = 0
                layer.y = 0
            self._array = np.array(layer.mask.convert("L"))
        else:
            self._array = np.array(layer.image.convert("RGBA"))
        self._selection = _local_selection(session, self._origin, self._array.shape[:2])
        self._active = True
        self._last = (event.x, event.y)
        self._stamp(session, event.x, event.y)
        self._publish()
        session.preview()

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._active or self._last is None:
            return
        radius, _, _ = _brush_params(session)
        spacing = max(1.0, radius * 0.35)
        points = interpolate(self._last, (event.x, event.y), spacing)
        for x, y in points:
            self._stamp(session, x, y)
        if points:
            self._last = points[-1]
        self._publish()
        session.preview()

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._active or self._layer is None:
            return
        self.pointer_move(session, event)
        layer = self._layer
        before = self._before
        created = self._created
        self._reset_stroke_buffers()
        name = "Erase" if self.erase else "Brush"
        session.finish_raster_edit(layer, before, created, name)

    def pointer_cancel(self, session) -> None:
        if self._layer is not None:
            session.cancel_raster_edit(self._layer, self._before, self._created)
        self._reset_stroke_buffers()

    def _reset_stroke_buffers(self) -> None:
        self._active = False
        self._array = None
        self._layer = None
        self._before = None
        self._last = None

    def _stamp(self, session, x: float, y: float) -> None:
        if self._array is None:
            return
        radius, hardness, opacity = _brush_params(session)
        local_x = x - self._origin[0]
        local_y = y - self._origin[1]
        if self._mask_mode:
            dab_mask(self._array, local_x, local_y, radius, hardness, opacity, self.erase, self._selection)
        else:
            dab(
                self._array,
                local_x,
                local_y,
                radius,
                hardness,
                session.foreground,
                opacity,
                self.erase,
                self._selection,
            )

    def _publish(self) -> None:
        if self._layer is None or self._array is None:
            return
        if self._mask_mode:
            self._layer.mask = Image.fromarray(self._array, "L")
        else:
            self._layer.image = Image.fromarray(self._array, "RGBA")


class BrushTool(_PaintTool):
    id = "brush"
    label = "Brush"
    shortcut = "B"
    hint = "Paint on the active layer. [ and ] change the size."


class EraserTool(_PaintTool):
    id = "eraser"
    label = "Eraser"
    shortcut = "E"
    erase = True
    hint = "Erase pixels, or lift a redaction mask when a redaction layer is active."


class TextTool(Tool):
    id = "text"
    label = "Text"
    shortcut = "T"
    cursor = "ibeam"
    hint = "Click to add text. Click an existing text layer to edit it."

    def pointer_down(self, session, event: PointerEvent) -> None:
        target = self._text_target(session, event)
        initial = target.text if target is not None else TextInfo(
            text="Text",
            font_path=default_font_path(),
            font_family="DejaVu Sans",
            size=48,
            color=session.foreground,
            rotation=0.0,
        )
        edited = session.prompt_text(initial)
        if edited is None or not edited.text.strip():
            return
        if target is None:
            session.add_text(edited, int(round(event.x)), int(round(event.y)))
        else:
            session.layer_index = session.document.index_of(target.id)
            session.update_text(target, edited)

    def _text_target(self, session, event: PointerEvent) -> Layer | None:
        hit = hit_opaque(session.document, event.x, event.y)
        if hit is not None and session.document.layers[hit].kind == "text":
            return session.document.layers[hit]
        active = session.active_layer()
        if active is not None and active.kind == "text":
            x, y, w, h = active.bounds()
            if x <= event.x <= x + w and y <= event.y <= y + h:
                return active
        return None


class RedactTool(Tool):
    id = "redact"
    label = "Redact"
    shortcut = "R"
    cursor = "cross"
    hint = "Draw a blur or pixelate mask. The pixels underneath stay intact until you export."

    def __init__(self) -> None:
        self._dragging = False
        self._start = (0.0, 0.0)
        self._points: list[tuple[float, float]] = []
        self._layer: Layer | None = None
        self._created = False
        self._before: dict | None = None

    def option_schema(self):
        return [
            {
                "key": "redact_shape",
                "label": "Shape",
                "type": "choice",
                "choices": ["rect", "ellipse", "freehand"],
                "default": "rect",
            },
            {
                "key": "redact_kind",
                "label": "Effect",
                "type": "choice",
                "choices": ["pixelate", "blur"],
                "default": "pixelate",
            },
            {
                "key": "redact_amount",
                "label": "Amount",
                "type": "int",
                "min": 1,
                "max": 80,
                "default": 12,
            },
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        # The mask is baked on release. Dragging only updates the rubber band,
        # because a live Gaussian blur of the whole canvas would stall the UI.
        self._dragging = True
        self._start = (event.x, event.y)
        self._points = [(event.x, event.y)]
        self._layer = None
        self._created = False
        self._before = None

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._points.append((event.x, event.y))
        session.preview()

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._points.append((event.x, event.y))
        self._dragging = False
        kind = str(session.option("redact_shape", "rect"))
        if kind != "freehand" and math.hypot(event.x - self._start[0], event.y - self._start[1]) < 2:
            self._points = []
            session.preview()
            return
        layer, created = session.ensure_redaction_layer()
        self._layer = layer
        self._created = created
        self._before = None if created else visual_state(layer)
        layer.redaction = RedactionInfo(
            str(session.option("redact_kind", "pixelate")),
            float(session.option("redact_amount", 12)),
        )
        self._preview(session, event)
        baked = self._layer
        before = self._before
        self._layer = None
        self._points = []
        if baked is None:
            return
        if created:
            try:
                index = session.document.layers.index(baked)
            except ValueError:
                index = len(session.document.layers)
            else:
                session.document.layers.pop(index)
            session.add_layer(baked, index=index, name="Redact")
            return
        if before is None:
            return
        new_state = visual_state(baked)
        apply_visual(baked, before)
        session.apply_layer_image(baked, new_state, "Redact", before=before)

    def pointer_cancel(self, session) -> None:
        self._dragging = False
        self._points = []
        self._layer = None
        session.preview()

    def _preview(self, session, event: PointerEvent) -> None:
        if self._layer is None:
            return
        doc = session.document
        shape = self._shape_mask(session, event)
        if self._before is not None and self._before["mask"] is not None:
            base = place_image(self._before["mask"], self._before["x"], self._before["y"], doc.width, doc.height, 0)
        else:
            base = Image.new("L", (doc.width, doc.height), 0)
        # Union, so a second stroke on the same redaction layer adds to the mask
        # instead of replacing regions the user already drew.
        combined = np.maximum(np.asarray(base), np.asarray(shape))
        self._layer.mask = Image.fromarray(combined, "L")
        self._layer.x = 0
        self._layer.y = 0

    def _shape_mask(self, session, event: PointerEvent) -> Image.Image:
        doc = session.document
        kind = str(session.option("redact_shape", "rect"))
        if kind == "freehand":
            return mask_freehand(doc.width, doc.height, self._points, radius=8)
        x0, y0 = self._start
        x1, y1 = event.x, event.y
        if kind == "ellipse":
            return mask_ellipse(doc.width, doc.height, x0, y0, x1, y1)
        return mask_rect(doc.width, doc.height, x0, y0, x1, y1)

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging or len(self._points) < 1:
            return []
        kind = str(session.option("redact_shape", "rect"))
        if kind == "freehand":
            return [Overlay(kind="polyline", points=list(self._points))]
        x0, y0 = self._start
        x1, y1 = self._points[-1]
        left, top = min(x0, x1), min(y0, y1)
        rect = (left, top, abs(x1 - x0), abs(y1 - y0))
        return [Overlay(kind="ellipse" if kind == "ellipse" else "rect", rect=rect)]


class SelectTool(Tool):
    id = "select"
    label = "Select"
    shortcut = "M"
    cursor = "cross"
    hint = "Drag a selection. Shift adds, Alt subtracts. Wand uses the tolerance."

    def __init__(self) -> None:
        self._dragging = False
        self._start = (0.0, 0.0)
        self._points: list[tuple[float, float]] = []

    def option_schema(self):
        return [
            {
                "key": "select_mode",
                "label": "Mode",
                "type": "choice",
                "choices": ["rect", "ellipse", "freehand", "wand"],
                "default": "rect",
            },
            {"key": "tolerance", "label": "Wand tolerance", "type": "int", "min": 0, "max": 255, "default": 32},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        if str(session.option("select_mode", "rect")) == "wand":
            self._wand(session, event)
            return
        self._dragging = True
        self._start = (event.x, event.y)
        self._points = [(event.x, event.y)]

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._points.append((event.x, event.y))
        session.preview()

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._points.append((event.x, event.y))
        mask = self._mask(session, event)
        self._dragging = False
        self._points = []
        mode = "replace"
        if event.shift:
            mode = "add"
        elif event.alt:
            mode = "subtract"
        combined = combine_selection(session.document.selection, mask, mode)
        session.set_selection(combined)

    def pointer_cancel(self, session) -> None:
        self._dragging = False
        self._points = []
        session.preview()

    def _mask(self, session, event: PointerEvent) -> Image.Image:
        doc = session.document
        kind = str(session.option("select_mode", "rect"))
        if kind == "freehand":
            return mask_freehand(doc.width, doc.height, self._points, radius=1)
        x0, y0 = self._start
        if kind == "ellipse":
            return mask_ellipse(doc.width, doc.height, x0, y0, event.x, event.y)
        return mask_rect(doc.width, doc.height, x0, y0, event.x, event.y)

    def _wand(self, session, event: PointerEvent) -> None:
        layer = session.active_layer()
        if layer is None or layer.kind == "redaction":
            session.show_message("Magic wand samples the active raster or text layer.")
            return
        local_x = int(math.floor(event.x - layer.x))
        local_y = int(math.floor(event.y - layer.y))
        local = connected_mask(layer.image, local_x, local_y, int(session.option("tolerance", 32)))
        canvas = Image.new("L", (session.document.width, session.document.height), 0)
        canvas.paste(local, (layer.x, layer.y))
        mode = "replace"
        if event.shift:
            mode = "add"
        elif event.alt:
            mode = "subtract"
        session.set_selection(combine_selection(session.document.selection, canvas, mode), name="Magic wand")

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging:
            return []
        kind = str(session.option("select_mode", "rect"))
        if kind == "freehand":
            return [Overlay(kind="polyline", points=list(self._points))]
        x0, y0 = self._start
        x1, y1 = self._points[-1] if self._points else self._start
        left, top = min(x0, x1), min(y0, y1)
        return [
            Overlay(
                kind="ellipse" if kind == "ellipse" else "rect",
                rect=(left, top, abs(x1 - x0), abs(y1 - y0)),
            )
        ]


class CropTool(Tool):
    id = "crop"
    label = "Crop"
    shortcut = "C"
    cursor = "cross"
    hint = "Drag a rectangle. The canvas crops when you release."

    def __init__(self) -> None:
        self._dragging = False
        self._start = (0.0, 0.0)
        self._end = (0.0, 0.0)

    def pointer_down(self, session, event: PointerEvent) -> None:
        self._dragging = True
        self._start = (event.x, event.y)
        self._end = (event.x, event.y)

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        session.preview()

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        self._dragging = False
        if math.hypot(self._end[0] - self._start[0], self._end[1] - self._start[1]) < 2:
            session.preview()
            return
        left, top, width, height = clamp_crop(
            session.document, self._start[0], self._start[1], self._end[0], self._end[1]
        )
        session.crop_to(left, top, width, height)

    def pointer_cancel(self, session) -> None:
        self._dragging = False
        session.preview()

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging:
            return []
        x0, y0 = self._start
        x1, y1 = self._end
        left, top = min(x0, x1), min(y0, y1)
        return [Overlay(kind="rect", rect=(left, top, abs(x1 - x0), abs(y1 - y0)))]


class FillTool(Tool):
    id = "fill"
    label = "Fill"
    shortcut = "G"
    cursor = "cross"
    hint = "Click to flood-fill similar connected pixels."

    def option_schema(self):
        return [
            {"key": "tolerance", "label": "Tolerance", "type": "int", "min": 0, "max": 255, "default": 32},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        layer, before, created = session.begin_raster_edit(paint_redaction=False)
        local_x = int(math.floor(event.x - layer.x))
        local_y = int(math.floor(event.y - layer.y))
        if not (0 <= local_x < layer.image.width and 0 <= local_y < layer.image.height):
            session.cancel_raster_edit(layer, before, created)
            return
        mask = connected_mask(layer.image, local_x, local_y, int(session.option("tolerance", 32)))
        selection = selection_for_layer(session.document, layer)
        if selection is not None:
            keep = np.minimum(np.asarray(mask), np.asarray(selection))
            mask = Image.fromarray(keep, "L")
        layer.image = fill_where(layer.image, mask, session.foreground)
        session.finish_raster_edit(layer, before, created, "Fill")


class EyedropperTool(Tool):
    id = "eyedropper"
    label = "Eyedrop"
    shortcut = "I"
    cursor = "cross"
    hint = "Click to pick a color from the flattened image."

    def pointer_down(self, session, event: PointerEvent) -> None:
        color = sample_color(composite(session.document), event.x, event.y)
        if color is None or color[3] == 0:
            session.show_message("Nothing opaque to sample there.")
            return
        session.foreground = color
        session.notify("color")


class ShapeTool(Tool):
    id = "shape"
    label = "Shape"
    shortcut = "U"
    cursor = "cross"
    hint = "Drag a rectangle, ellipse, or line. Shift is not required."

    def __init__(self) -> None:
        self._dragging = False
        self._start = (0.0, 0.0)
        self._end = (0.0, 0.0)

    def option_schema(self):
        return [
            {
                "key": "shape_kind",
                "label": "Shape",
                "type": "choice",
                "choices": ["rect", "ellipse", "line"],
                "default": "rect",
            },
            {"key": "stroke_width", "label": "Stroke", "type": "int", "min": 1, "max": 64, "default": 4},
            {"key": "shape_filled", "label": "Fill shape", "type": "bool", "default": False},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        self._dragging = True
        self._start = (event.x, event.y)
        self._end = (event.x, event.y)

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        session.preview()

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        self._dragging = False
        if math.hypot(self._end[0] - self._start[0], self._end[1] - self._start[1]) < 1:
            session.preview()
            return
        layer, before, created = session.begin_raster_edit(paint_redaction=False)
        kind = str(session.option("shape_kind", "rect"))
        selection = selection_for_layer(session.document, layer)
        # Gestures arrive in canvas coordinates. The bitmap is local to the layer.
        ox, oy = float(layer.x), float(layer.y)
        image = draw_shape(
            layer.image,
            kind,
            (self._start[0] - ox, self._start[1] - oy, self._end[0] - ox, self._end[1] - oy),
            session.foreground,
            int(session.option("stroke_width", 4)),
            bool(session.option("shape_filled", False)),
            selection,
            line=((self._start[0] - ox, self._start[1] - oy), (self._end[0] - ox, self._end[1] - oy)),
        )
        layer.image = image
        session.finish_raster_edit(layer, before, created, "Shape")

    def pointer_cancel(self, session) -> None:
        self._dragging = False
        session.preview()

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging:
            return []
        kind = str(session.option("shape_kind", "rect"))
        if kind == "line":
            return [Overlay(kind="polyline", points=[self._start, self._end])]
        x0, y0 = self._start
        x1, y1 = self._end
        left, top = min(x0, x1), min(y0, y1)
        return [
            Overlay(
                kind="ellipse" if kind == "ellipse" else "rect",
                rect=(left, top, abs(x1 - x0), abs(y1 - y0)),
            )
        ]


class HandTool(Tool):
    id = "hand"
    label = "Hand"
    shortcut = "H"
    cursor = "hand"
    hint = "Drag to pan. You can also hold Space with any tool."


class ZoomTool(Tool):
    id = "zoom"
    label = "Zoom"
    shortcut = "Z"
    cursor = "cross"
    hint = "Click to zoom in. Alt-click zooms out. The wheel zooms at the cursor."
