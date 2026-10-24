"""Blank-canvas templates: screen pixels and physical paper sizes.

Paper sizes stay in millimetres or inches. Pixels are computed at the chosen
resolution with the same rounding as the image-size dialog, so an A4 page at
300 dpi is the printed sheet and not a 72 dpi canvas that only looks large.
"""

from __future__ import annotations

from dataclasses import dataclass

from pixedit.geom import pixels_from_inches

# 25.4 millimetres to the inch. Paper presets store ISO sizes in millimetres
# and convert through this so the pixel count matches a physical page.
MM_PER_INCH = 25.4

SCREEN_DPI = 72.0
PAPER_DPI = 300.0
RESOLUTIONS = (72, 150, 300)
CUSTOM_WIDTH = 1280
CUSTOM_HEIGHT = 800


@dataclass(frozen=True)
class CanvasPreset:
    """One row in the New canvas template list.

    ``kind`` is ``custom``, ``screen``, or ``paper``. Screen presets carry a
    fixed pixel size. Paper presets carry a portrait page, either in
    millimetres (ISO) or inches (US Letter). Custom has neither.
    """

    id: str
    label: str
    kind: str
    width_px: int = 0
    height_px: int = 0
    width_mm: float | None = None
    height_mm: float | None = None
    width_in: float | None = None
    height_in: float | None = None


PRESETS: tuple[CanvasPreset, ...] = (
    CanvasPreset("custom", "Custom (1280 × 800)", "custom"),
    CanvasPreset("hd", "HD (1920 × 1080)", "screen", width_px=1920, height_px=1080),
    CanvasPreset("uhd", "4K (3840 × 2160)", "screen", width_px=3840, height_px=2160),
    CanvasPreset("square", "Square (1080 × 1080)", "screen", width_px=1080, height_px=1080),
    CanvasPreset("a4", "A4", "paper", width_mm=210, height_mm=297),
    CanvasPreset("a3", "A3", "paper", width_mm=297, height_mm=420),
    CanvasPreset("a5", "A5", "paper", width_mm=148, height_mm=210),
    CanvasPreset("letter", "US Letter", "paper", width_in=8.5, height_in=11),
)

_BY_ID = {preset.id: preset for preset in PRESETS}


def preset_by_id(preset_id: str) -> CanvasPreset:
    try:
        return _BY_ID[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown canvas template {preset_id!r}") from exc


def preset_size(preset: str, orientation: str, dpi: float) -> tuple[int, int, float]:
    """Pixel width, height, and the dpi to store on the document.

    Paper sizes convert the physical page at ``dpi``. Portrait keeps the
    stored side order, which is the short side across for these sheets.
    Landscape swaps the sides. Screen sizes ignore orientation and dpi and
    stay at 72. Custom is the usual 1280 × 800 canvas at 72 dpi.
    """
    item = preset_by_id(preset)
    if item.kind == "custom":
        return CUSTOM_WIDTH, CUSTOM_HEIGHT, SCREEN_DPI
    if item.kind == "screen":
        return item.width_px, item.height_px, SCREEN_DPI
    if orientation not in ("portrait", "landscape"):
        raise ValueError(f"Unknown orientation {orientation!r}")
    # A non-positive resolution cannot describe a page. Fall back to screen
    # dpi rather than dividing the page into a zero-pixel edge.
    stored = float(dpi)
    if stored <= 0:
        stored = SCREEN_DPI
    width_in, height_in = _page_inches(item)
    if orientation == "landscape":
        width_in, height_in = height_in, width_in
    return pixels_from_inches(width_in, stored), pixels_from_inches(height_in, stored), stored


def page_label(preset: str, orientation: str) -> str | None:
    """Physical page, such as ``210 × 297 mm``. None for screen and custom."""
    item = preset_by_id(preset)
    if item.kind != "paper":
        return None
    if orientation not in ("portrait", "landscape"):
        raise ValueError(f"Unknown orientation {orientation!r}")
    if item.width_mm is not None and item.height_mm is not None:
        width, height, unit = item.width_mm, item.height_mm, "mm"
    else:
        width, height, unit = float(item.width_in or 0), float(item.height_in or 0), "in"
    if orientation == "landscape":
        width, height = height, width
    return f"{_measure(width)} × {_measure(height)} {unit}"


def _page_inches(preset: CanvasPreset) -> tuple[float, float]:
    if preset.width_mm is not None and preset.height_mm is not None:
        return preset.width_mm / MM_PER_INCH, preset.height_mm / MM_PER_INCH
    return float(preset.width_in or 0), float(preset.height_in or 0)


def _measure(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.4f}".rstrip("0").rstrip(".")
