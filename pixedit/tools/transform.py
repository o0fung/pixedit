"""Move, scale, and rotate the selected layers on the canvas."""

from __future__ import annotations

import math

from pixedit.geom import (
    _corner_of,
    _orbit_point,
    _place_from_corner,
    clip_box,
    hit_layer,
    top_layer_at,
    layer_union_bounds,
    placed_box,
    rotate_visual_around,
    scale_visual_in_box,
    size_from_anchor,
    union_boxes,
)
from pixedit.guides import snap_delta, snap_point
from pixedit.model import (
    Layer,
    apply_visual,
    visual_state,
)
from pixedit.proxy import DragProxy, flatten_for_drag
from pixedit.tools.base import (
    Overlay,
    PointerEvent,
    Tool,
)

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
        # Set for a large picture. The layer itself stays full resolution
        # until the handle is released.
        self._stand_in: DragProxy | None = None
        self._live_box: tuple[float, float, float, float] | None = None
        self._proxy_active = False

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
            top = top_layer_at(session.document, event.x, event.y)
            if top is not None and not any(item is top for item in session.document.layers):
                session.focus_layer(top)
            else:
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
        # One screen-sized flatten. Move and scale draw it; release resamples.
        self._stand_in = flatten_for_drag(layers, session._stack(), self._start_box, event.zoom)
        self._live_box = self._start_box
        self._proxy_active = False

    def drag_proxy(self) -> DragProxy | None:
        """The stand-in the canvas draws, or None when this drag resamples."""
        if not self._proxy_active:
            return None
        return self._stand_in

    def pointer_move(self, session, event: PointerEvent) -> None:
        if self._mode is None or not self._starts or self._start_box is None:
            return
        if self._stand_in is not None:
            # The bitmap stays at full resolution. The canvas moves the flatten.
            self._proxy_active = True
            self._place_stand_in(session, event)
            session.notify("overlay")
            return
        self._apply(session, event, final=False)
        rows = self._rows(session, self._ids, self._starts, self._origins)
        session.preview(self._changed_box(session, rows))

    def _changed_box(self, session, rows) -> tuple[int, int, int, int] | None:
        """Document rectangle covering where the layers were and where they are."""
        doc = session.document
        box = None
        for layer, start, _origin in rows:
            box = union_boxes(box, placed_box(layer, doc.width, doc.height))
            image = start["image"]
            box = union_boxes(
                box,
                clip_box(
                    (int(start["x"]), int(start["y"]), int(start["x"]) + image.width, int(start["y"]) + image.height),
                    doc.width,
                    doc.height,
                ),
            )
        return box

    def _move_delta(self, session, event: PointerEvent) -> tuple[float, float]:
        dx = event.x - self._start_mouse[0]
        dy = event.y - self._start_mouse[1]
        group = session.document.group
        return snap_delta(
            dx,
            dy,
            self._start_box,
            group.guides_v,
            group.guides_h,
            bool(session.option("snap", True)),
        )

    def _scale_event(self, session, event: PointerEvent) -> PointerEvent:
        group = session.document.group
        sx, sy = snap_point(
            event.x,
            event.y,
            group.guides_v,
            group.guides_h,
            bool(session.option("snap", True)),
        )
        return PointerEvent(sx, sy, event.shift, event.alt, event.ctrl, event.button, event.zoom)

    def _scale_factors(self, event: PointerEvent) -> tuple[float, float]:
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
        return new_w / max(box[2], 1e-6), new_h / max(box[3], 1e-6)

    def _scaled_union(self, event: PointerEvent) -> tuple[int, int, int, int]:
        """Where the mouse-down box lands. Same corner math as one layer."""
        assert self._start_box is not None
        box = self._start_box
        scale_x, scale_y = self._scale_factors(event)
        target_w = max(1, int(round(box[2] * scale_x)))
        target_h = max(1, int(round(box[3] * scale_y)))
        corner = _corner_of(box[0], box[1], box[2], box[3], self._anchor_name)
        ax, ay = self._anchor
        mapped = (ax + (corner[0] - ax) * scale_x, ay + (corner[1] - ay) * scale_y)
        return _place_from_corner(self._anchor_name, mapped, target_w, target_h) + (target_w, target_h)

    def _turn_delta(self, event: PointerEvent) -> float:
        assert self._start_box is not None
        x, y, width, height = self._start_box
        cx = x + width / 2.0
        cy = y + height / 2.0
        angle = math.degrees(math.atan2(event.y - cy, event.x - cx))
        # Screen Y grows downward, so atan2 increases clockwise. Wrap so the
        # gesture doesn't jump at ±180.
        return (angle - self._start_angle + 180.0) % 360.0 - 180.0

    def _place_stand_in(self, session, event: PointerEvent) -> None:
        """Move the flatten. The layer bitmaps are not touched."""
        assert self._stand_in is not None and self._start_box is not None
        if self._mode == "move":
            dx, dy = self._move_delta(session, event)
            x, y, width, height = self._start_box
            # One translation for the whole flatten. Integer origins share it
            # except exactly on a half pixel, which the release resolves per layer.
            ix = int(round(x + dx)) - int(round(x))
            iy = int(round(y + dy)) - int(round(y))
            self._stand_in.live_box = (x + ix, y + iy, width, height)
            self._stand_in.angle = 0.0
        elif self._mode in _FIXED:
            placed = self._scaled_union(self._scale_event(session, event))
            self._stand_in.live_box = placed
            self._stand_in.angle = 0.0
        elif self._mode == "rot":
            self._stand_in.live_box = self._start_box
            self._stand_in.angle = self._turn_delta(event)
        self._live_box = self._stand_in.live_box

    def _scale_group(
        self,
        rows: list[tuple[Layer, dict, tuple[int, int]]],
        event: PointerEvent,
        *,
        final: bool,
    ) -> None:
        scale_x, scale_y = self._scale_factors(event)
        for layer, start, _origin in rows:
            apply_visual(
                layer,
                scale_visual_in_box(
                    start,
                    self._anchor,
                    self._anchor_name,
                    scale_x,
                    scale_y,
                    zoom=event.zoom,
                    final=final,
                ),
            )

    def _rotate_group(
        self,
        rows: list[tuple[Layer, dict, tuple[int, int]]],
        event: PointerEvent,
        *,
        final: bool,
    ) -> None:
        assert self._start_box is not None
        x, y, width, height = self._start_box
        # The frame turns by this angle and ignores the bitmap bounds, which
        # grow into a straight box as the rectangle turns.
        delta = self._turn_delta(event)
        self._angle = delta
        center = (x + width / 2.0, y + height / 2.0)
        for layer, start, _origin in rows:
            apply_visual(layer, rotate_visual_around(start, delta, center, zoom=event.zoom, final=final))

    def _apply(self, session, event: PointerEvent, *, final: bool) -> None:
        """Move, scale, or rotate. ``final`` resamples at full resolution."""
        rows = self._rows(session, self._ids, self._starts, self._origins)
        if self._mode == "move":
            dx, dy = self._move_delta(session, event)
            for layer, _start, origin in rows:
                layer.x = int(round(origin[0] + dx))
                layer.y = int(round(origin[1] + dy))
        elif self._mode in _FIXED:
            self._scale_group(rows, self._scale_event(session, event), final=final)
        elif self._mode == "rot":
            self._rotate_group(rows, event, final=final)

    def pointer_up(self, session, event: PointerEvent) -> None:
        if self._mode is None or not self._starts:
            return
        mode = self._mode
        ids = list(self._ids)
        starts = list(self._starts)
        origins = list(self._origins)
        moved = math.hypot(event.x - self._start_mouse[0], event.y - self._start_mouse[1])
        if moved < 0.5:
            # A large drag never wrote the layer, so there is nothing to copy back.
            if self._stand_in is None:
                rows = self._rows(session, ids, starts, origins)
                for layer, start, _origin in rows:
                    apply_visual(layer, start)
            self._clear()
            session.preview()
            return
        # The drag showed a stand-in when the picture was large. Release writes
        # the Lanczos or bicubic pixels, or the integer position for a move.
        showed_proxy = self._proxy_active
        self._apply(session, event, final=True)
        self._clear()
        rows = self._rows(session, ids, starts, origins)
        before = len(session.history.undo_stack)
        if mode == "move":
            session.commit_layer_moves(
                [(layer, layer.x, layer.y, origin[0], origin[1]) for layer, _start, origin in rows]
            )
        else:
            session.commit_layer_visuals(
                [(layer, visual_state(layer), start) for layer, start, _origin in rows],
                "Transform",
            )
        # A move that lands on the same pixel commits nothing, so the canvas
        # would keep painting the stand-in. The cached picture is still correct.
        if showed_proxy and len(session.history.undo_stack) == before:
            session.notify("overlay")

    def pointer_cancel(self, session) -> bool:
        # A click that misses clears the mode but leaves the snapshots, so
        # Escape can still put a half-finished drag back.
        active = bool(self._starts)
        if active and self._stand_in is None:
            for layer, start, _origin in self._rows(session, self._ids, self._starts, self._origins):
                apply_visual(layer, start)
        # Drop the stand-in before the repaint, or the canvas would keep drawing it.
        self._clear()
        if active:
            session.preview()
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
        # The stored bounds have not moved yet. The frame follows the flatten.
        if self._proxy_active and self._live_box is not None:
            box = self._live_box
        else:
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
        self._stand_in = None
        self._live_box = None
        self._proxy_active = False
