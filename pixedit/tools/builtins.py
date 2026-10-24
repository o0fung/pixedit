"""Built-in canvas tools.

Tools mutate the session only through its command helpers. A gesture that
previews (brush, transform, redaction) snapshots the layer first and commits
a single undo step on release. Escape calls ``pointer_cancel``.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from pixedit.composite import composite
from pixedit.geom import (
    _orbit_point,
    clip_box,
    combine_selection,
    grow_layer_to_cover,
    hit_layer,
    layer_union_bounds,
    mask_ellipse,
    mask_freehand,
    mask_rect,
    paint_bounds,
    place_image,
    rotate_visual_around,
    scale_visual_in_box,
    selection_for_layer,
    size_from_anchor,
)
from pixedit.model import Layer, apply_visual, visual_state
from pixedit.paint import (
    BrushFootprint,
    catmull_segment,
    draw_shape,
    fill_document,
    paint_footprints,
    paint_mask_footprints,
    reflect_point,
    sample_color,
    stroke_spacing,
    wand_document,
)
from pixedit.tools.base import Overlay, PointerEvent, Tool

_FIXED = {"nw": "se", "ne": "sw", "sw": "ne", "se": "nw"}

# Screen diagonals. Top-left and bottom-right share one arrow; the other two
# corners share the opposite diagonal. The box that scales is axis-aligned.
_HANDLE_CURSOR = {
    "nw": "size-fdiag",
    "se": "size-fdiag",
    "ne": "size-bdiag",
    "sw": "size-bdiag",
    "rot": "rotate",
    "move": "size",
}


def _union_box(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return (min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3]))


def _brush_params(session) -> tuple[float, float, float]:
    # Size is the brush diameter in image pixels, matching the usual editor convention.
    radius = max(0.5, float(session.option("brush_size", 24)) / 2.0)
    hardness = float(session.option("brush_hardness", 85)) / 100.0
    opacity = float(session.option("brush_opacity", 100)) / 100.0
    return radius, hardness, opacity


def _region_cursor(kind: str) -> str:
    """Pointer for a rectangle, ellipse, freehand loop, or magic wand."""
    if kind == "ellipse":
        return "select-ellipse"
    if kind == "freehand":
        return "lasso"
    if kind == "wand":
        return "wand"
    return "select-rect"


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


class TransformTool(Tool):
    id = "transform"
    label = "Transform"
    shortcut = "V"
    cursor = "size"
    hint = (
        "Drag to move the selected layers, even when the pointer starts on another "
        "layer or outside their box. Corners scale, the handle above rotates. "
        "Shift-click adds a layer. Command-click or Ctrl-click toggles one. "
        "Buttons flip and turn the selection."
    )

    def panel_actions(self):
        # new_row keeps Scale and Rotate together under the quarter-turns,
        # even when the column is wide enough for all six icons.
        return [
            {"id": "flip_active_horizontal", "label": "Flip horizontal", "icon": "flip-h"},
            {"id": "flip_active_vertical", "label": "Flip vertical", "icon": "flip-v"},
            {"id": "rotate_active_cw", "label": "Rotate 90° clockwise", "icon": "rotate-cw"},
            {"id": "rotate_active_ccw", "label": "Rotate 90° counter-clockwise", "icon": "rotate-ccw"},
            {"id": "scale_layer", "label": "Scale…", "icon": "scale", "new_row": True},
            {"id": "rotate_layer", "label": "Rotate…", "icon": "rotate"},
        ]

    def __init__(self) -> None:
        self._mode: str | None = None
        self._ids: list[str] = []
        self._starts: list[dict] = []
        self._origins: list[tuple[int, int]] = []
        self._start_mouse = (0.0, 0.0)
        self._anchor_name = ""
        self._anchor = (0.0, 0.0)
        self._start_angle = 0.0
        self._angle = 0.0
        self._start_box: tuple[float, float, float, float] | None = None

    def pointer_down(self, session, event: PointerEvent) -> None:
        # The layers already selected are the only ones that move. Where the
        # pointer starts does not pick a different layer.
        # 1. A handle scales or rotates that selection, and does not change it.
        # 2. Shift adds the layer under the pointer. Command or Ctrl toggles
        #    it. A plain click leaves the selection alone.
        # 3. The drag then moves whatever is selected, including a start on
        #    another layer or outside the box.
        layers = session.selected_layers()
        if layers:
            handle = self._hit_handle(layer_union_bounds(layers), event)
            if handle is not None:
                self._begin(session, handle, event)
                return
        if event.ctrl or event.shift:
            hit = hit_layer(session.document, event.x, event.y)
            if hit is not None:
                self._apply_click_selection(session, hit, event)
        if not session.selected_layers():
            self._mode = None
            return
        self._begin(session, "move", event)

    def _apply_click_selection(self, session, hit: int, event: PointerEvent) -> None:
        """Edit the selection from a Shift or Command/Ctrl click.

        Shift adds the layer under the pointer. Command or Ctrl toggles it,
        and the last remaining layer stays selected. A plain click never
        reaches here, so it cannot swap in the layer under the pointer.
        """
        layer = session.document.layers[hit]
        before = (tuple(session.selected_ids()), session.layer_index)
        selected = list(before[0])
        selected_set = set(selected)
        if event.ctrl:
            if layer.id in selected_set and len(selected_set) > 1:
                remaining = [layer_id for layer_id in selected if layer_id != layer.id]
                active = session.active_layer()
                active_id = active.id if active is not None and active.id in remaining else None
                session.select_layers(remaining, active_id=active_id)
            elif layer.id not in selected_set:
                session.select_layers([*selected, layer.id], active_id=layer.id)
        elif event.shift:
            session.select_layers([*selected, layer.id], active_id=layer.id)
        after = (tuple(session.selected_ids()), session.layer_index)
        if after != before:
            session.notify("change")

    def _begin(self, session, mode: str, event: PointerEvent) -> None:
        layers = session.selected_layers()
        if not layers:
            self._clear()
            return
        self._mode = mode
        self._ids = [layer.id for layer in layers]
        self._starts = [visual_state(layer) for layer in layers]
        self._origins = [(layer.x, layer.y) for layer in layers]
        self._start_mouse = (event.x, event.y)
        self._start_box = layer_union_bounds(layers)
        self._angle = 0.0
        if mode in _FIXED:
            self._anchor_name = _FIXED[mode]
            self._anchor = self._handle_points(self._start_box, event.zoom)[self._anchor_name]
        elif mode == "rot":
            x, y, width, height = self._start_box
            self._start_angle = math.degrees(
                math.atan2(event.y - (y + height / 2.0), event.x - (x + width / 2.0))
            )

    def pointer_move(self, session, event: PointerEvent) -> None:
        if self._mode is None or not self._starts or self._start_box is None:
            return
        rows = self._rows(session, self._ids, self._starts, self._origins)
        if self._mode == "move":
            dx = event.x - self._start_mouse[0]
            dy = event.y - self._start_mouse[1]
            for layer, _start, origin in rows:
                layer.x = int(round(origin[0] + dx))
                layer.y = int(round(origin[1] + dy))
        elif self._mode in _FIXED:
            self._scale_group(rows, event)
        elif self._mode == "rot":
            self._rotate_group(rows, event)
        session.preview()

    def _scale_group(self, rows: list[tuple[Layer, dict, tuple[int, int]]], event: PointerEvent) -> None:
        assert self._start_box is not None
        box = self._start_box
        _left, _top, width, height = size_from_anchor(
            self._anchor,
            (event.x, event.y),
            lock_aspect=event.shift,
            aspect=box[2] / max(box[3], 1e-6),
        )
        new_w = max(1, int(round(width)))
        new_h = max(1, int(round(height)))
        scale_x = new_w / max(box[2], 1e-6)
        scale_y = new_h / max(box[3], 1e-6)
        for layer, start, _origin in rows:
            apply_visual(layer, scale_visual_in_box(start, self._anchor, self._anchor_name, scale_x, scale_y))

    def _rotate_group(self, rows: list[tuple[Layer, dict, tuple[int, int]]], event: PointerEvent) -> None:
        assert self._start_box is not None
        x, y, width, height = self._start_box
        cx = x + width / 2.0
        cy = y + height / 2.0
        angle = math.degrees(math.atan2(event.y - cy, event.x - cx))
        # Screen Y grows downward, so atan2 increases clockwise. Wrap so the
        # gesture doesn't jump at ±180. Same rule as a single layer.
        # The frame turns by this angle and ignores the bitmap bounds, which
        # grow into a straight box as the rectangle turns.
        delta = (angle - self._start_angle + 180.0) % 360.0 - 180.0
        self._angle = delta
        center = (cx, cy)
        for layer, start, _origin in rows:
            apply_visual(layer, rotate_visual_around(start, delta, center))

    def pointer_up(self, session, event: PointerEvent) -> None:
        if self._mode is None or not self._starts:
            return
        self.pointer_move(session, event)
        mode = self._mode
        ids = list(self._ids)
        starts = list(self._starts)
        origins = list(self._origins)
        moved = math.hypot(event.x - self._start_mouse[0], event.y - self._start_mouse[1])
        self._clear()
        rows = self._rows(session, ids, starts, origins)
        if moved < 0.5:
            for layer, start, _origin in rows:
                apply_visual(layer, start)
            session.preview()
            return
        if mode == "move":
            session.commit_layer_moves(
                [(layer, layer.x, layer.y, origin[0], origin[1]) for layer, _start, origin in rows]
            )
            return
        session.commit_layer_visuals(
            [(layer, visual_state(layer), start) for layer, start, _origin in rows],
            "Transform",
        )

    def pointer_cancel(self, session) -> bool:
        # A click that misses clears the mode but leaves the snapshots, so
        # Escape can still put a half-finished drag back.
        active = bool(self._starts)
        if active:
            for layer, start, _origin in self._rows(session, self._ids, self._starts, self._origins):
                apply_visual(layer, start)
            session.preview()
        self._clear()
        return active

    def overlays(self, session) -> list[Overlay]:
        layers = session.selected_layers()
        if not layers:
            return []
        zoom = float(getattr(session, "view_zoom", 1.0) or 1.0)
        # During a turn, the bitmap's straight box grows. The frame stays the
        # mouse-down size and only rotates, so the handle stays with the cursor.
        if self._mode == "rot" and self._start_box is not None:
            return [self._rotated_overlay(self._start_box, self._angle, zoom)]
        box = layer_union_bounds(layers)
        points = self._handle_points(box, zoom)
        x, y, width, height = box
        handles = [(name, px, py) for name, (px, py) in points.items()]
        return [Overlay(kind="handles", rect=(x, y, width, height), handles=handles)]

    def _rotated_overlay(
        self, box: tuple[float, float, float, float], degrees: float, zoom: float
    ) -> Overlay:
        """Turn the mouse-down box in place. Edge lengths do not change.

        The live bitmap is a straight box around the turned pixels, so it
        grows and shrinks with the angle. Drawing that box would scale the
        frame and pull the rotate handle off the cursor. Corners and the
        handle orbit the same center, with the same clockwise screen rule as
        the pixels.
        """
        x, y, width, height = box
        cx = x + width / 2.0
        cy = y + height / 2.0
        corners = {
            "nw": (x, y),
            "ne": (x + width, y),
            "se": (x + width, y + height),
            "sw": (x, y + height),
        }
        turned = {name: _orbit_point(px, py, cx, cy, degrees) for name, (px, py) in corners.items()}
        # Outward from the top edge. At 0° that is straight up, negative Y.
        lift = 28.0 / max(zoom, 0.05)
        ox, oy = _orbit_point(0.0, -lift, 0.0, 0.0, degrees)
        top_x = (turned["nw"][0] + turned["ne"][0]) / 2.0
        top_y = (turned["nw"][1] + turned["ne"][1]) / 2.0
        order = ("nw", "ne", "se", "sw")
        return Overlay(
            kind="handles",
            points=[turned[name] for name in order],
            handles=[(name, turned[name][0], turned[name][1]) for name in order]
            + [("rot", top_x + ox, top_y + oy)],
        )

    def _handle_points(
        self, box: tuple[float, float, float, float], zoom: float
    ) -> dict[str, tuple[float, float]]:
        x, y, width, height = box
        lift = 28.0 / max(zoom, 0.05)
        return {
            "nw": (x, y),
            "ne": (x + width, y),
            "sw": (x, y + height),
            "se": (x + width, y + height),
            "rot": (x + width / 2.0, y - lift),
        }

    def hover_cursor(self, session, event: PointerEvent) -> str:
        """Pointer for the handle under the cursor, or the move arrows.

        A drag keeps the pointer for the action that started, so leaving the
        handle does not fall back to the move arrows until the button is up.
        The grab distance is the same one a click uses.
        """
        if self._mode == "rot":
            return "rotate"
        if self._mode in _HANDLE_CURSOR:
            return _HANDLE_CURSOR[self._mode]
        layers = session.selected_layers()
        if not layers:
            return "size"
        handle = self._hit_handle(layer_union_bounds(layers), event)
        if handle is None:
            return "size"
        return _HANDLE_CURSOR.get(handle, "size")

    def _hit_handle(self, box: tuple[float, float, float, float], event: PointerEvent) -> str | None:
        limit = 8.0 / max(event.zoom, 0.05)
        best = None
        best_dist = limit
        for name, (hx, hy) in self._handle_points(box, event.zoom).items():
            dist = math.hypot(event.x - hx, event.y - hy)
            if dist <= best_dist:
                best = name
                best_dist = dist
        return best

    def _rows(
        self,
        session,
        ids: list[str],
        starts: list[dict],
        origins: list[tuple[int, int]],
    ) -> list[tuple[Layer, dict, tuple[int, int]]]:
        rows = []
        for layer_id, start, origin in zip(ids, starts, origins):
            try:
                layer = session.document.layer_by_id(layer_id)
            except KeyError:
                continue
            rows.append((layer, start, origin))
        return rows

    def _clear(self) -> None:
        self._mode = None
        self._ids = []
        self._starts = []
        self._origins = []
        self._start_box = None
        self._angle = 0.0


class _PaintTool(Tool):
    """Shared stroke loop for the brush and the eraser."""

    erase = False
    cursor = "brush"

    def __init__(self) -> None:
        self._active = False
        self._layer: Layer | None = None
        self._before: dict | None = None
        self._created = False
        self._mask_mode = False
        self._array: np.ndarray | None = None
        self._selection: np.ndarray | None = None
        self._origin = (0, 0)
        self._raw: list[tuple[float, float]] = []
        # Index of the raw point where the provisional tip starts. Everything
        # before that index already has its outgoing segment committed.
        self._final_until = 0
        self._tip_patch: np.ndarray | None = None
        self._tip_rect: tuple[int, int, int, int] | None = None
        self._tip_canvas: tuple[int, int, int, int] | None = None
        # The pointer-down dab is a full coat so a click is visible at once.
        # The first move puts it back and repaints with gained stamps.
        self._click_patch: np.ndarray | None = None
        self._click_rect: tuple[int, int, int, int] | None = None
        self._click_canvas: tuple[int, int, int, int] | None = None
        self._footprint: BrushFootprint | None = None
        self._footprint_key: tuple[float, float, float] | None = None

    def wants_stroke_sampling(self, _session) -> bool:
        return True

    def stroke_live(self) -> bool:
        return self._active

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
        self._raw = [(event.x, event.y)]
        self._final_until = 0
        self._tip_patch = None
        self._tip_rect = None
        self._tip_canvas = None
        self._click_patch = None
        self._click_rect = None
        self._click_canvas = None
        self._footprint = None
        self._footprint_key = None
        self._ensure_cover(session, event.x, event.y)
        # Full coat, not the thinned stroke stamp, so a click matches the preview.
        self._remember_click(session, event.x, event.y)
        rect = self._stamp_points(session, [(event.x, event.y)], gained=False)
        self._publish(rect)
        session.preview(rect)

    def pointer_move(self, session, event: PointerEvent) -> None:
        """Paint one new coordinate immediately. Tests and plugins use this."""
        self.extend_points(session, [(event.x, event.y)])

    def extend_points(self, session, points: list[tuple[float, float]]) -> None:
        """Add raw coordinates and paint them as one update.

        The stroke is a centripetal Catmull-Rom through the raw points. A
        segment needs one point of lookahead, so the last segment is only a
        tip until the next batch arrives:

        1. On the first move, put back the pointer-down dab. That dab is the
           full coat so a click shows immediately. The stroke then paints only
           spacing-gained stamps, so the start is one coat rather than the
           click stacked on the line.
        2. Restore the previous tip (its shape changes once lookahead exists).
           Both restores happen before the bitmap can grow, so the saved
           patches still line up.
        3. Commit every segment that now has a real point ahead of it.
        4. Save the pixels under the new tip, then stamp that tip. Mouse-up
           keeps the tip; it is already the end of the stroke.

        A new segment starts after the point the previous one stamped, so a
        translucent brush is not darkened by redrawing the tip. One publish
        and one preview cover the whole batch.
        """
        if not self._active or self._array is None or not self._take_points(points):
            return
        # Drop the click before any growth. Later moves have nothing saved here.
        dirty = self._restore_click()
        dirty = _union_box(dirty, self._restore_tip())
        final_points, tip_points = self._new_curve_points(session)
        self._cover_points(session, final_points)
        self._cover_points(session, tip_points)
        if final_points:
            dirty = _union_box(dirty, self._stamp_points(session, final_points, gained=True))
        if tip_points:
            self._save_tip(session, tip_points)
            dirty = _union_box(dirty, self._tip_canvas)
            dirty = _union_box(dirty, self._stamp_points(session, tip_points, gained=True))
        self._publish(dirty)
        session.preview(dirty)

    def commit_stroke(self, session, _event: PointerEvent | None = None) -> None:
        """Keep the current tip and store the stroke as one undo step.

        The canvas passes the release pointer. The last coordinate was already
        flushed, so the brush does not read it.
        """
        if not self._active or self._layer is None:
            return
        layer = self._layer
        before = self._before
        created = self._created
        self._reset_stroke_buffers()
        name = "Erase" if self.erase else "Brush"
        session.finish_raster_edit(layer, before, created, name)

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._active or self._layer is None:
            return
        self.extend_points(session, [(event.x, event.y)])
        self.commit_stroke(session)

    def pointer_cancel(self, session) -> bool:
        active = self._layer is not None
        if active:
            session.cancel_raster_edit(self._layer, self._before, self._created)
        self._reset_stroke_buffers()
        return active

    def _reset_stroke_buffers(self) -> None:
        self._active = False
        self._array = None
        self._layer = None
        self._before = None
        self._raw = []
        self._final_until = 0
        self._tip_patch = None
        self._tip_rect = None
        self._tip_canvas = None
        self._click_patch = None
        self._click_rect = None
        self._click_canvas = None
        self._footprint = None
        self._footprint_key = None

    def _take_points(self, points: list[tuple[float, float]]) -> bool:
        added = False
        for x, y in points:
            point = (float(x), float(y))
            if self._raw:
                last_x, last_y = self._raw[-1]
                if math.hypot(point[0] - last_x, point[1] - last_y) < 0.05:
                    continue
            self._raw.append(point)
            added = True
        return added

    def _new_curve_points(self, session) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Split the uncommitted curve into a final run and the new tip."""
        count = len(self._raw)
        if count < 2:
            return [], []
        radius, hardness, _ = _brush_params(session)
        spacing = stroke_spacing(radius, hardness)
        final: list[tuple[float, float]] = []
        # The segment that ends on the newest point stays provisional: the
        # spline still needs a point ahead of it.
        for index in range(self._final_until, count - 2):
            final.extend(self._segment_points(index, spacing))
        self._final_until = count - 2
        return final, self._segment_points(count - 2, spacing)

    def _segment_points(self, index: int, spacing: float) -> list[tuple[float, float]]:
        start = self._raw[index]
        end = self._raw[index + 1]
        before = self._raw[index - 1] if index > 0 else reflect_point(start, end)
        after = self._raw[index + 2] if index + 2 < len(self._raw) else reflect_point(end, start)
        return catmull_segment(before, start, end, after, spacing)

    def _footprint_for(self, session) -> BrushFootprint:
        radius, hardness, opacity = _brush_params(session)
        key = (radius, hardness, opacity)
        if self._footprint is None or self._footprint_key != key:
            self._footprint = BrushFootprint(radius, hardness, opacity)
            self._footprint_key = key
        return self._footprint

    def _cover_points(self, session, points: list[tuple[float, float]]) -> None:
        for x, y in points:
            self._ensure_cover(session, x, y)

    def _stamp_points(self, session, points: list[tuple[float, float]], *, gained: bool):
        if self._array is None or not points:
            return None
        footprint = self._footprint_for(session)
        if self._mask_mode:
            return paint_mask_footprints(
                self._array,
                self._origin,
                points,
                footprint,
                self.erase,
                self._selection,
                gained=gained,
            )
        return paint_footprints(
            self._array,
            self._origin,
            points,
            footprint,
            session.foreground,
            self.erase,
            self._selection,
            gained=gained,
        )

    def _remember_click(self, session, x: float, y: float) -> None:
        """Save the pixels under the pointer-down dab before it is painted."""
        box = self._local_stamp_box(session, [(x, y)])
        if box is None or self._array is None:
            return
        x0, y0, x1, y1 = box
        self._click_rect = box
        self._click_patch = self._array[y0:y1, x0:x1].copy()
        ox, oy = self._origin
        self._click_canvas = (x0 + ox, y0 + oy, x1 + ox, y1 + oy)

    def _restore_click(self):
        """Put back the pixels under the pointer-down dab. Returns its canvas box."""
        canvas = self._click_canvas
        if self._click_patch is None or self._array is None or self._click_rect is None:
            self._click_patch = None
            self._click_rect = None
            self._click_canvas = None
            return None
        x0, y0, x1, y1 = self._click_rect
        height, width = self._array.shape[:2]
        matches = self._click_patch.shape[:2] == (y1 - y0, x1 - x0)
        inside = 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        if matches and inside:
            self._array[y0:y1, x0:x1] = self._click_patch
        self._click_patch = None
        self._click_rect = None
        self._click_canvas = None
        return canvas

    def _local_stamp_box(self, session, points: list[tuple[float, float]]):
        """Local bitmap box covered by ``points``, or None when nothing lands."""
        if self._array is None or not points:
            return None
        footprint = self._footprint_for(session)
        box = None
        height, width = self._array.shape[:2]
        ox, oy = self._origin
        for cx, cy in points:
            placed = footprint.place(cx, cy)
            kernel = placed[2]
            local_x = placed[0] - ox
            local_y = placed[1] - oy
            x0 = max(0, local_x)
            y0 = max(0, local_y)
            x1 = min(width, local_x + kernel.shape[1])
            y1 = min(height, local_y + kernel.shape[0])
            if x0 < x1 and y0 < y1:
                box = _union_box(box, (x0, y0, x1, y1))
        return box

    def _restore_tip(self):
        """Put back the pixels under the provisional tip. Returns its canvas box."""
        canvas = self._tip_canvas
        if self._tip_patch is None or self._array is None or self._tip_rect is None:
            self._tip_patch = None
            self._tip_rect = None
            self._tip_canvas = None
            return None
        x0, y0, x1, y1 = self._tip_rect
        height, width = self._array.shape[:2]
        matches = self._tip_patch.shape[:2] == (y1 - y0, x1 - x0)
        inside = 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        if matches and inside:
            self._array[y0:y1, x0:x1] = self._tip_patch
        self._tip_patch = None
        self._tip_rect = None
        self._tip_canvas = None
        return canvas

    def _save_tip(self, session, points: list[tuple[float, float]]) -> None:
        """Remember the bitmap under the tip before those stamps land."""
        box = self._local_stamp_box(session, points)
        if box is None or self._array is None:
            return
        x0, y0, x1, y1 = box
        self._tip_rect = box
        self._tip_patch = self._array[y0:y1, x0:x1].copy()
        ox, oy = self._origin
        self._tip_canvas = (x0 + ox, y0 + oy, x1 + ox, y1 + oy)

    def _ensure_cover(self, session, x: float, y: float) -> None:
        """Grow the stroke buffer so a dab outside the bitmap still lands.

        Redaction masks stay at their current size (document-sized when the
        mask tool created them). Raster growth is clipped to the document,
        with slack so the next moves in a stroke reuse the same buffer.
        """
        if self._array is None or self._layer is None or self._mask_mode:
            return
        radius, _, _ = _brush_params(session)
        rad = int(math.ceil(radius)) + 1
        ix, iy = int(math.floor(x)), int(math.floor(y))
        dab = (ix - rad, iy - rad, ix + rad + 1, iy + rad + 1)
        height, width = self._array.shape[:2]
        ox, oy = self._origin
        if dab[0] >= ox and dab[1] >= oy and dab[2] <= ox + width and dab[3] <= oy + height:
            return
        empty = not bool(np.any(self._array[..., 3]))
        bounds = paint_bounds(
            self._origin,
            (width, height),
            dab,
            session.document.width,
            session.document.height,
            slack=32,
            empty=empty,
        )
        if bounds is None:
            return
        nx0, ny0, nx1, ny1 = bounds
        new_w, new_h = nx1 - nx0, ny1 - ny0
        if (nx0, ny0) == (ox, oy) and new_w == width and new_h == height:
            return
        grown = np.zeros((new_h, new_w, 4), np.uint8)
        if not empty:
            paste_x = ox - nx0
            paste_y = oy - ny0
            grown[paste_y : paste_y + height, paste_x : paste_x + width] = self._array
        self._array = grown
        self._origin = (nx0, ny0)
        self._layer.x = nx0
        self._layer.y = ny0
        self._selection = _local_selection(session, self._origin, self._array.shape[:2])

    def _publish(self, canvas_rect) -> None:
        """Copy changed pixels into the layer. The whole buffer is copied only when its size changed."""
        if self._layer is None or self._array is None:
            return
        mode = "L" if self._mask_mode else "RGBA"
        current = self._layer.mask if self._mask_mode else self._layer.image
        height, width = self._array.shape[:2]
        local = None if canvas_rect is None else self._local_from_canvas(canvas_rect)
        size_changed = current is None or current.size != (width, height)
        if size_changed or local is None:
            # fromarray shares memory with the buffer. Copy so a later stamp
            # cannot rewrite the image the undo snapshot will read.
            image = Image.fromarray(np.ascontiguousarray(self._array), mode).copy()
        else:
            x0, y0, x1, y1 = local
            patch = Image.fromarray(np.ascontiguousarray(self._array[y0:y1, x0:x1]), mode)
            current.paste(patch, (x0, y0))
            image = current
        if self._mask_mode:
            self._layer.mask = image
        else:
            self._layer.image = image

    def _local_from_canvas(self, box) -> tuple[int, int, int, int] | None:
        if self._array is None:
            return None
        ox, oy = self._origin
        height, width = self._array.shape[:2]
        x0 = max(0, int(math.floor(box[0])) - ox)
        y0 = max(0, int(math.floor(box[1])) - oy)
        x1 = min(width, int(math.ceil(box[2])) - ox)
        y1 = min(height, int(math.ceil(box[3])) - oy)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1


class BrushTool(_PaintTool):
    id = "brush"
    label = "Brush"
    shortcut = "B"
    hint = "Paint on the active layer. The cursor matches the diameter, and the panel shows the stamp. [ and ] change the size."


class EraserTool(_PaintTool):
    id = "eraser"
    label = "Eraser"
    shortcut = "E"
    erase = True
    hint = "Erase pixels, or lift a mask when a mask layer is active. The panel shows the hole. [ and ] change the size."


class TextTool(Tool):
    id = "text"
    label = "Text"
    shortcut = "T"
    cursor = "ibeam"
    hint = "Click inside a text box to edit it. Click empty space to add text."

    def option_schema(self):
        return [
            {"key": "text_font", "label": "Font", "type": "font"},
            {"key": "text_style", "label": "Style", "type": "font_style"},
            {
                "key": "text_align",
                "label": "Align",
                "type": "choice",
                "icons": True,
                "choices": [
                    ("left", "Left", "align-left"),
                    ("center", "Center", "align-center"),
                    ("right", "Right", "align-right"),
                    ("justify", "Justify", "align-justify"),
                ],
                "default": "left",
            },
            {"key": "text_size", "label": "Size", "type": "spin", "min": 1, "max": 512, "default": 48},
            {
                "key": "text_line_spacing",
                "label": "Spacing",
                "type": "int",
                "min": 0,
                "max": 80,
                "default": 4,
                "suffix": " px",
            },
            {
                "key": "text_rotation",
                "label": "Rotation",
                "type": "int",
                "min": -180,
                "max": 180,
                "default": 0,
                "suffix": "°",
            },
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        target = self._text_target(session, event)
        if target is None or target.text is None:
            initial = session.text_info_from_options("Text")
        else:
            # Load this layer into the panel before the word dialog. The dialog
            # keeps that style, so a click does not restyle it from the previous
            # panel values. Cancel still leaves the layer selected.
            session.layer_index = session.document.index_of(target.id)
            session.adopt_text_style(target.text)
            session.notify("change")
            initial = target.text
        edited = session.prompt_text(initial)
        if edited is None or not edited.text.strip():
            return
        if target is None:
            session.add_text(edited, int(round(event.x)), int(round(event.y)))
        else:
            session.layer_index = session.document.index_of(target.id)
            session.update_text(target, edited)

    def _text_target(self, session, event: PointerEvent) -> Layer | None:
        # The topmost box wins, so a picture stacked above a text layer is not
        # edited by a click on the picture. A gap in the text box still hits
        # the text, because that layer sits above the background.
        hit = hit_layer(session.document, event.x, event.y)
        if hit is None:
            return None
        layer = session.document.layers[hit]
        if layer.kind != "text":
            return None
        return layer


class MaskTool(Tool):
    id = "mask"
    label = "Mask"
    shortcut = "R"
    cursor = "select-rect"
    hint = "Draw a mask on the active layer. The effect stays separate until you apply it, merge, or export."

    def __init__(self) -> None:
        self._dragging = False
        # Latched at press. The cursor poll keeps running until release even
        # if the mode control changes during the drag.
        self._freehand = False
        self._start = (0.0, 0.0)
        self._points: list[tuple[float, float]] = []
        self._layer: Layer | None = None
        self._created = False
        self._before: dict | None = None

    def option_schema(self):
        from pixedit.model import MASK_EFFECTS

        return [
            {
                "key": "mask_shape",
                "label": "Shape",
                "type": "choice",
                "icons": True,
                "choices": [
                    ("rect", "Rectangle", "geom-rect"),
                    ("ellipse", "Ellipse", "geom-ellipse"),
                    ("freehand", "Freehand", "geom-freehand"),
                ],
                "default": "rect",
            },
            {
                "key": "mask_kind",
                "label": "Effect",
                "type": "choice",
                "choices": [
                    ("pixelate", "Pixelate"),
                    ("blur", "Blur"),
                    ("solid", "Solid"),
                    ("noise", "Noise"),
                    ("desaturate", "Desaturate"),
                    ("invert", "Invert"),
                    ("dim", "Dim"),
                    ("sharpen", "Sharpen"),
                ],
            "default": "pixelate",
            "amounts": MASK_EFFECTS,
            "amount_key": "mask_amount",
        },
        {
            "key": "mask_amount",
            "label": "Block",
            "type": "int",
            "min": 2,
            "max": 64,
            "default": 12,
        },
    ]

    def cursor_name(self, session) -> str:
        # Same glyphs as Select. The gesture is the region, not the effect.
        return _region_cursor(str(session.option("mask_shape", "rect")))

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
        kind = str(session.option("mask_shape", "rect"))
        if kind != "freehand" and math.hypot(event.x - self._start[0], event.y - self._start[1]) < 2:
            self._points = []
            session.preview()
            return
        layer, created = session.ensure_redaction_layer()
        self._layer = layer
        self._created = created
        self._before = None if created else visual_state(layer)
        layer.redaction = session.current_mask_info(layer.redaction)
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
            session.add_layer(baked, index=index, name="Mask")
            return
        if before is None:
            return
        new_state = visual_state(baked)
        apply_visual(baked, before)
        session.apply_layer_image(baked, new_state, "Mask", before=before)

    def pointer_cancel(self, session) -> bool:
        active = self._dragging or self._layer is not None
        self._dragging = False
        self._points = []
        self._layer = None
        if active:
            session.preview()
        return active

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
        kind = str(session.option("mask_shape", "rect"))
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
        kind = str(session.option("mask_shape", "rect"))
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
    shortcut = "S"
    cursor = "select-rect"
    hint = (
        "Drag a selection. Ctrl+X cuts it to a new layer, Ctrl+C copies it, Delete clears pixels, Esc deselects. "
        "Wand also selects empty space outside the layer, and stops at the canvas edge."
    )

    def __init__(self) -> None:
        self._dragging = False
        # Locked at press time. The poll keeps running until release even if
        # the mode menu changes during the drag.
        self._freehand = False
        self._start = (0.0, 0.0)
        self._points: list[tuple[float, float]] = []

    def option_schema(self):
        return [
            {
                "key": "select_mode",
                "label": "Mode",
                "type": "choice",
                "icons": True,
                "choices": [
                    ("rect", "Rectangular", "select-rect"),
                    ("ellipse", "Elliptical", "select-ellipse"),
                    ("freehand", "Freehand", "select-freehand"),
                    ("wand", "Magic wand", "geom-wand"),
                ],
                "default": "rect",
            },
            {"key": "tolerance", "label": "Wand tolerance", "type": "int", "min": 0, "max": 255, "default": 32},
        ]

    def cursor_name(self, session) -> str:
        return _region_cursor(str(session.option("select_mode", "rect")))

    def wants_stroke_sampling(self, session) -> bool:
        return str(session.option("select_mode", "rect")) == "freehand"

    def stroke_live(self) -> bool:
        return self._dragging and self._freehand

    def panel_actions(self):
        return [
            {"id": "select_all", "label": "Select all", "shortcut": "Ctrl+A", "icon": "select-all"},
            {"id": "deselect", "label": "Deselect", "shortcut": "Ctrl+D", "needs_selection": True, "icon": "deselect"},
            {"id": "invert_selection", "label": "Invert", "shortcut": "Ctrl+Shift+I", "needs_selection": True, "icon": "select-invert"},
            {"id": "cut_selection_to_layer", "label": "Cut to layer", "shortcut": "Ctrl+X", "needs_selection": True, "icon": "cut-to-layer"},
            {"id": "copy_selection_to_layer", "label": "Copy to layer", "shortcut": "Ctrl+C", "needs_selection": True, "icon": "copy-to-layer"},
            {"id": "clear_selection_contents", "label": "Clear pixels", "shortcut": "Delete", "needs_selection": True, "icon": "clear-pixels"},
            {"id": "crop_to_selection", "label": "Crop to selection", "needs_selection": True, "icon": "crop-to-selection"},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        if str(session.option("select_mode", "rect")) == "wand":
            self._wand(session, event)
            return
        self._dragging = True
        self._freehand = str(session.option("select_mode", "rect")) == "freehand"
        self._start = (event.x, event.y)
        self._points = [(event.x, event.y)]

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        if self._freehand:
            self.extend_points(session, [(event.x, event.y)])
            return
        self._points.append((event.x, event.y))
        session.preview()

    def extend_points(self, session, points: list[tuple[float, float]]) -> None:
        """Append freehand samples. The canvas polls the cursor and flushes batches here.

        A full preview would rebuild the photo and stall that poll, so a new
        point only asks the view to redraw the rubber band. A point within
        0.05 image pixels of the last one is dropped, the same rule as the brush.
        """
        if not self._dragging or not self._freehand:
            return
        added = False
        for x, y in points:
            point = (float(x), float(y))
            if self._points:
                last_x, last_y = self._points[-1]
                if math.hypot(point[0] - last_x, point[1] - last_y) < 0.05:
                    continue
            self._points.append(point)
            added = True
        if added:
            session.notify("overlay")

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        if self._freehand:
            self.extend_points(session, [(event.x, event.y)])
            self.commit_stroke(session, event)
            return
        self._points.append((event.x, event.y))
        self._apply_selection(session, self._mask(session, event), event)

    def commit_stroke(self, session, event: PointerEvent | None = None) -> None:
        """Close the lasso from the stored points.

        The canvas flushes the last samples before this call. The points are
        the polygon vertices. Shift adds to the selection and Alt subtracts.
        """
        if not self._dragging or not self._freehand:
            return
        doc = session.document
        mask = mask_freehand(doc.width, doc.height, self._points, radius=1)
        self._apply_selection(session, mask, event)

    def _apply_selection(self, session, mask: Image.Image, event: PointerEvent | None) -> None:
        self._dragging = False
        self._freehand = False
        self._points = []
        mode = "replace"
        if event is not None:
            if event.shift:
                mode = "add"
            elif event.alt:
                mode = "subtract"
        session.set_selection(combine_selection(session.document.selection, mask, mode))

    def pointer_cancel(self, session) -> bool:
        active = self._dragging
        self._dragging = False
        self._freehand = False
        self._points = []
        if active:
            session.preview()
        return active

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
        # A click anywhere on the canvas is a seed, including empty space the
        # layer bitmap does not cover. Outside the canvas: explain and do not
        # change the selection. The plate matches Fill, so the walk stops at a
        # different pixel or the canvas edge. Shift and Alt still combine
        # afterwards; an existing selection does not clip the walk.
        doc = session.document
        px = int(math.floor(event.x))
        py = int(math.floor(event.y))
        if not (0 <= px < doc.width and 0 <= py < doc.height):
            session.show_message("Click is outside the canvas.")
            return
        canvas = wand_document(
            layer.image,
            layer.x,
            layer.y,
            doc.width,
            doc.height,
            px,
            py,
            int(session.option("tolerance", 32)),
        )
        mode = "replace"
        if event.shift:
            mode = "add"
        elif event.alt:
            mode = "subtract"
        session.set_selection(combine_selection(doc.selection, canvas, mode), name="Magic wand")

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging:
            return []
        # The gesture keeps the shape it started with. A mode change while
        # the button is down does not turn a lasso into a rectangle.
        if self._freehand:
            return [Overlay(kind="polyline", points=list(self._points))]
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
    label = "Sizes & Crop"
    shortcut = "C"
    cursor = "crop"
    hint = (
        "Drag a rectangle. Release crops the canvas, and a drag past the edge expands it. "
        "Shift keeps the proportion."
    )

    def __init__(self) -> None:
        self._dragging = False
        self._start = (0.0, 0.0)
        self._end = (0.0, 0.0)
        self._overlay_rect: tuple[int, int, int, int] | None = None

    def option_schema(self):
        return [
            {
                "key": "crop_ratio",
                "label": "Proportion",
                "type": "choice",
                "choices": [
                    ("free", "Free"),
                    ("original", "Original"),
                    ("1:1", "1:1"),
                    ("4:3", "4:3"),
                    ("16:9", "16:9"),
                    ("custom", "Custom"),
                ],
                "default": "free",
            },
            {"key": "crop_ratio_w", "label": "Ratio width", "type": "spin", "min": 1, "max": 10000, "default": 1},
            {"key": "crop_ratio_h", "label": "Ratio height", "type": "spin", "min": 1, "max": 10000, "default": 1},
        ]

    def panel_actions(self):
        return [
            {"id": "canvas_size", "label": "Canvas size…"},
            {"id": "image_size", "label": "Image size…"},
            {"id": "trim", "label": "Trim transparent edges"},
            {"id": "crop_to_selection", "label": "Crop to selection", "needs_selection": True},
        ]

    def _locked_aspect(self, session, event: PointerEvent) -> float | None:
        """Width/height to lock, or None for a free rectangle.

        A chosen proportion always locks. Shift while Free is selected locks
        the canvas proportion, the same way Transform's Shift locks the layer.
        """
        choice = str(session.option("crop_ratio", "free"))
        if choice == "1:1":
            return 1.0
        if choice == "4:3":
            return 4.0 / 3.0
        if choice == "16:9":
            return 16.0 / 9.0
        if choice == "custom":
            width = max(1, int(session.option("crop_ratio_w", 1)))
            height = max(1, int(session.option("crop_ratio_h", 1)))
            return width / height
        if choice == "original" or event.shift:
            document = session.document
            return document.width / max(1, document.height)
        return None

    def _crop_box(self, session, event: PointerEvent) -> tuple[int, int, int, int]:
        aspect = self._locked_aspect(session, event)
        left, top, width, height = size_from_anchor(
            self._start,
            (event.x, event.y),
            lock_aspect=aspect is not None,
            aspect=aspect or 1.0,
        )
        return (
            int(round(left)),
            int(round(top)),
            max(1, int(round(width))),
            max(1, int(round(height))),
        )

    def pointer_down(self, session, event: PointerEvent) -> None:
        self._dragging = True
        self._start = (event.x, event.y)
        self._end = (event.x, event.y)
        self._overlay_rect = self._crop_box(session, event)

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        self._overlay_rect = self._crop_box(session, event)
        session.preview()

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        self._dragging = False
        box = self._crop_box(session, event)
        self._overlay_rect = None
        if math.hypot(self._end[0] - self._start[0], self._end[1] - self._start[1]) < 2:
            session.preview()
            return
        session.crop_to(*box)

    def pointer_cancel(self, session) -> bool:
        active = self._dragging
        self._dragging = False
        self._overlay_rect = None
        if active:
            session.preview()
        return active

    def overlays(self, session) -> list[Overlay]:
        if not self._dragging or self._overlay_rect is None:
            return []
        left, top, width, height = self._overlay_rect
        return [Overlay(kind="rect", rect=(left, top, width, height))]


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


class ColorTool(Tool):
    id = "color"
    label = "Image & Eyedrop"
    shortcut = "I"
    cursor = "eyedrop"
    hint = "Click to pick a color. Sliders adjust the active layer; Apply keeps the change."

    def option_schema(self):
        return [
            {"key": "color_hue", "label": "Hue", "type": "int", "min": -180, "max": 180, "default": 0, "suffix": "°", "live": "color"},
            {"key": "color_saturation", "label": "Saturation", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
            {"key": "color_brightness", "label": "Brightness", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
            {"key": "color_contrast", "label": "Contrast", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
            {"key": "color_highlight", "label": "Highlight", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
            {"key": "color_shade", "label": "Shade", "type": "int", "min": -100, "max": 100, "default": 0, "live": "color"},
        ]

    def panel_actions(self):
        return [
            {"id": "apply_color", "label": "Apply", "needs_color_preview": True},
            {"id": "reset_color", "label": "Reset", "needs_color_preview": True},
        ]

    def pointer_down(self, session, event: PointerEvent) -> None:
        color = sample_color(composite(session.document), event.x, event.y)
        if color is None or color[3] == 0:
            session.show_message("Nothing opaque to sample there.")
            return
        session.foreground = color
        session.notify("color")

    def pointer_cancel(self, session) -> bool:
        return bool(session.cancel_color_adjust())


class ShapeTool(Tool):
    id = "shape"
    label = "Shape"
    shortcut = "U"
    cursor = "shape-rect"
    hint = "Drag a rectangle, ellipse, or line. The canvas shows the shape, fill, and stroke while you drag."

    def __init__(self) -> None:
        self._dragging = False
        self._start = (0.0, 0.0)
        self._end = (0.0, 0.0)
        self._layer: Layer | None = None
        self._before: dict | None = None
        self._created = False
        self._snap_image: Image.Image | None = None
        self._snap_xy = (0, 0)

    def option_schema(self):
        return [
            {
                "key": "shape_kind",
                "label": "Shape",
                "type": "choice",
                "icons": True,
                "choices": [
                    ("rect", "Rectangle", "geom-rect"),
                    ("ellipse", "Ellipse", "geom-ellipse"),
                    ("line", "Line", "geom-line"),
                ],
                "default": "rect",
            },
            {"key": "stroke_width", "label": "Stroke", "type": "int", "min": 1, "max": 64, "default": 4},
            {"key": "shape_filled", "label": "Fill shape?", "type": "bool", "default": False},
        ]

    def cursor_name(self, session) -> str:
        kind = str(session.option("shape_kind", "rect"))
        if kind == "ellipse":
            return "shape-ellipse"
        if kind == "line":
            return "shape-line"
        return "shape-rect"

    def pointer_down(self, session, event: PointerEvent) -> None:
        self._cancel_edit(session)
        self._dragging = True
        self._start = (event.x, event.y)
        self._end = (event.x, event.y)

    def pointer_move(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        if math.hypot(self._end[0] - self._start[0], self._end[1] - self._start[1]) < 1:
            return
        self._show_preview(session)

    def pointer_up(self, session, event: PointerEvent) -> None:
        if not self._dragging:
            return
        self._end = (event.x, event.y)
        self._dragging = False
        if math.hypot(self._end[0] - self._start[0], self._end[1] - self._start[1]) < 1:
            self._cancel_edit(session)
            session.preview()
            return
        box = self._canvas_box(session)
        if box is None:
            self._cancel_edit(session)
            session.show_message("The shape is outside the canvas.")
            session.preview()
            return
        # The preview pass paints with the same draw used at commit time, so
        # release does not change fill, stroke, color, or alpha.
        self._show_preview(session)
        layer, before, created = self._take_edit()
        if layer is None:
            return
        session.finish_raster_edit(layer, before, created, "Shape")

    def pointer_cancel(self, session) -> bool:
        active = self._dragging or self._layer is not None
        self._dragging = False
        self._cancel_edit(session)
        return active

    def _ensure_edit(self, session) -> None:
        if self._layer is not None:
            return
        layer, before, created = session.begin_raster_edit(paint_redaction=False)
        self._layer = layer
        self._before = before
        self._created = created
        # Snapshot after text is rasterized. Each move restores this and
        # redraws, so dragging does not stack copies of the shape.
        self._snap_image = layer.image.copy()
        self._snap_xy = (layer.x, layer.y)

    def _restore_snap(self) -> None:
        if self._layer is None or self._snap_image is None:
            return
        self._layer.image = self._snap_image.copy()
        self._layer.x, self._layer.y = self._snap_xy

    def _paint(self, session, box: tuple[int, int, int, int]) -> None:
        layer = self._layer
        if layer is None:
            return
        # Grow to the stroke's canvas box first so a small layer can hold it.
        # finish_raster_edit then crops back to the opaque pixels.
        grow_layer_to_cover(layer, box)
        kind = str(session.option("shape_kind", "rect"))
        selection = selection_for_layer(session.document, layer)
        ox, oy = float(layer.x), float(layer.y)
        layer.image = draw_shape(
            layer.image,
            kind,
            (self._start[0] - ox, self._start[1] - oy, self._end[0] - ox, self._end[1] - oy),
            session.foreground,
            int(session.option("stroke_width", 4)),
            bool(session.option("shape_filled", False)),
            selection,
            line=((self._start[0] - ox, self._start[1] - oy), (self._end[0] - ox, self._end[1] - oy)),
        )

    def _show_preview(self, session) -> None:
        self._ensure_edit(session)
        self._restore_snap()
        box = self._canvas_box(session)
        if box is None:
            session.preview()
            return
        self._paint(session, box)
        session.preview()

    def _take_edit(self) -> tuple[Layer | None, dict | None, bool]:
        layer, before, created = self._layer, self._before, self._created
        self._layer = None
        self._before = None
        self._created = False
        self._snap_image = None
        return layer, before, created

    def _cancel_edit(self, session) -> None:
        layer, before, created = self._take_edit()
        if layer is None:
            return
        session.cancel_raster_edit(layer, before, created)

    def _canvas_box(self, session) -> tuple[int, int, int, int] | None:
        """Half-open document box covering the shape, including its stroke."""
        stroke = int(session.option("stroke_width", 4))
        pad = max(1, stroke)
        x0 = math.floor(min(self._start[0], self._end[0])) - pad
        y0 = math.floor(min(self._start[1], self._end[1])) - pad
        x1 = math.ceil(max(self._start[0], self._end[0])) + pad + 1
        y1 = math.ceil(max(self._start[1], self._end[1])) + pad + 1
        return clip_box((int(x0), int(y0), int(x1), int(y1)), session.document.width, session.document.height)


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
    cursor = "zoom-in"
    hint = "Click to zoom in. Alt-click zooms out. The wheel zooms at the cursor."

    def panel_actions(self):
        return [
            {"id": "zoom_in", "label": "Zoom in", "shortcut": "Ctrl++", "icon": "zoom"},
            {"id": "zoom_out", "label": "Zoom out", "shortcut": "Ctrl+-", "icon": "zoom-out"},
            {"id": "zoom_fit", "label": "Fit", "shortcut": "Ctrl+0", "icon": "zoom-fit"},
            {"id": "zoom_100", "label": "100%", "shortcut": "Ctrl+1", "icon": "zoom-100"},
        ]
