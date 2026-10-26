"""Page guides and the snap used by crop, transform, and shapes.

Guides are pixel positions on one page. Vertical guides are x, horizontal
guides are y. Snap is a few pixels, and the caller can turn it off.
"""

from __future__ import annotations

SNAP_DISTANCE = 6.0


def snap_point(
    x: float,
    y: float,
    vertical: tuple[int, ...] | list[int],
    horizontal: tuple[int, ...] | list[int],
    enabled: bool,
) -> tuple[float, float]:
    """Pull ``(x, y)`` onto the nearest guide within ``SNAP_DISTANCE``."""
    if not enabled:
        return float(x), float(y)
    return _snap_axis(float(x), vertical), _snap_axis(float(y), horizontal)


def snap_delta(
    dx: float,
    dy: float,
    box: tuple[float, float, float, float] | None,
    vertical: tuple[int, ...] | list[int],
    horizontal: tuple[int, ...] | list[int],
    enabled: bool,
) -> tuple[float, float]:
    """Adjust a move so a box edge or its center lands on a guide.

    ``box`` is ``(x, y, width, height)`` before the move. Left, right, and
    the horizontal center compete; the closest guide within range wins.
    The same rule runs for the top, bottom, and vertical center.
    """
    if not enabled or box is None:
        return float(dx), float(dy)
    x, y, width, height = box
    left = float(x) + float(dx)
    right = left + float(width)
    top = float(y) + float(dy)
    bottom = top + float(height)
    dx = float(dx) + _correction((left, right, (left + right) / 2.0), vertical)
    dy = float(dy) + _correction((top, bottom, (top + bottom) / 2.0), horizontal)
    return dx, dy


def shift_guides(
    vertical: tuple[int, ...] | list[int],
    horizontal: tuple[int, ...] | list[int],
    left: int,
    top: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Move guides by a crop window. The window origin becomes (0, 0)."""
    return tuple(int(value) - int(left) for value in vertical), tuple(int(value) - int(top) for value in horizontal)


def scale_guides(
    vertical: tuple[int, ...] | list[int],
    horizontal: tuple[int, ...] | list[int],
    old_w: int,
    old_h: int,
    new_w: int,
    new_h: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Scale guide positions with a resampled canvas."""

    def scale(values: tuple[int, ...] | list[int], old: int, new: int) -> tuple[int, ...]:
        if old <= 0:
            return tuple(int(value) for value in values)
        return tuple(int(round(int(value) * int(new) / int(old))) for value in values)

    return scale(vertical, old_w, new_w), scale(horizontal, old_h, new_h)


def rotate_guides_cw(
    vertical: tuple[int, ...] | list[int],
    horizontal: tuple[int, ...] | list[int],
    height: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Guides after a 90° clockwise canvas turn.

    A vertical line at ``x`` becomes a horizontal line at ``y = x``.
    A horizontal line at ``y`` becomes a vertical line at ``x = height - 1 - y``.
    That is the same map Pillow's ROTATE_270 applies to a pixel.
    """
    new_vertical = tuple(int(height) - 1 - int(value) for value in horizontal)
    new_horizontal = tuple(int(value) for value in vertical)
    return new_vertical, new_horizontal


def rotate_guides_ccw(
    vertical: tuple[int, ...] | list[int],
    horizontal: tuple[int, ...] | list[int],
    width: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Guides after a 90° counter-clockwise canvas turn."""
    new_vertical = tuple(int(value) for value in horizontal)
    new_horizontal = tuple(int(width) - 1 - int(value) for value in vertical)
    return new_vertical, new_horizontal


def flip_guides(
    vertical: tuple[int, ...] | list[int],
    horizontal: tuple[int, ...] | list[int],
    width: int,
    height: int,
    *,
    horizontal_flip: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Mirror guides with a canvas flip. ``horizontal_flip`` mirrors left-right."""
    if horizontal_flip:
        return tuple(int(width) - 1 - int(value) for value in vertical), tuple(int(value) for value in horizontal)
    return tuple(int(value) for value in vertical), tuple(int(height) - 1 - int(value) for value in horizontal)


def _snap_axis(value: float, guides: tuple[int, ...] | list[int]) -> float:
    best: float | None = None
    best_dist = SNAP_DISTANCE + 1e-6
    for guide in guides:
        dist = abs(value - float(guide))
        if dist <= SNAP_DISTANCE and dist < best_dist:
            best = float(guide)
            best_dist = dist
    return value if best is None else best


def _correction(edges: tuple[float, ...], guides: tuple[int, ...] | list[int]) -> float:
    best = 0.0
    best_dist = SNAP_DISTANCE + 1e-6
    found = False
    for edge in edges:
        for guide in guides:
            dist = abs(edge - float(guide))
            if dist <= SNAP_DISTANCE and dist < best_dist:
                best = float(guide) - edge
                best_dist = dist
                found = True
    return best if found else 0.0
