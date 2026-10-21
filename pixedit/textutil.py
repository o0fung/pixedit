"""Font discovery and text rasterization.

Rendering stays on Pillow so tests and plugins do not need a display. The UI
lists the same font files in the text dialog.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pixedit.model import TextInfo

_FONT_ROOTS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / ".local/share/fonts"),
    "C:/Windows/Fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
)


@lru_cache(maxsize=1)
def discover_fonts() -> tuple[tuple[str, str, str], ...]:
    """Return ``(family, style, path)`` for installed TrueType/OpenType files."""
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for root in _FONT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                lower = name.lower()
                if not lower.endswith((".ttf", ".otf", ".ttc")):
                    continue
                # Coverage fonts such as the full Noto set make the text dialog
                # unusable. Keep the common cuts and every other family.
                if "noto" in lower and not any(
                    token in lower for token in ("notosans-regular", "notoserif-regular", "notosansmono")
                ):
                    continue
                path = os.path.join(dirpath, name)
                if path in seen:
                    continue
                seen.add(path)
                stem = Path(name).stem.replace("-", " ").replace("_", " ")
                family, _, style = stem.partition(" ")
                found.append((family or stem, style or "Regular", path))
    found.sort(key=lambda item: (item[0].lower(), item[1].lower()))
    return tuple(found)


def default_font_path() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    fonts = discover_fonts()
    return fonts[0][2] if fonts else None


def load_font(path: str | None, size: int) -> ImageFont.ImageFont:
    size = max(1, int(size))
    if path and os.path.isfile(path):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_text(info: TextInfo) -> Image.Image:
    """Rasterize ``info`` to a tight RGBA image.

    Rotation is clockwise, matching the rest of the editor. Pillow itself
    rotates counter-clockwise, so the angle is negated after the glyphs are drawn.
    """
    font = load_font(info.font_path, info.size)
    text = info.text if info.text else " "
    probe = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(probe)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    left, top, right, bottom = bbox
    width = max(1, int(right - left) + 8)
    height = max(1, int(bottom - top) + 8)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill = info.color if len(info.color) == 4 else (*info.color[:3], 255)
    draw.multiline_text((-left + 4, -top + 4), text, font=font, fill=fill, spacing=4)
    if abs(info.rotation) > 0.01:
        image = image.rotate(-float(info.rotation), resample=Image.Resampling.BICUBIC, expand=True)
    return image
