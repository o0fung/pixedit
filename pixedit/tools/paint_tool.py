"""Shared brush stroke used by the brush and the eraser.

A drag previews dabs on the layer. Release commits one undo step. Escape
puts the layer back.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from pixedit.geom import (
    paint_bounds,
    place_image,
)
from pixedit.model import Layer
from pixedit.paint import (
    BrushFootprint,
    catmull_segment,
    paint_footprints,
    paint_mask_footprints,
    reflect_point,
    stroke_spacing,
)
from pixedit.tools.base import (
    PointerEvent,
    Tool,
)


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
