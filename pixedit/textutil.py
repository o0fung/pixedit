"""Font discovery and text rasterization.

Rendering stays on Pillow so tests and plugins do not need a display. The UI
lists the same font files in the text options panel.
"""

from __future__ import annotations

import math
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
                # Coverage fonts such as the full Noto set make the font menu
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


_REGULAR_STYLES = {"regular", "normal", "roman", "book"}


def _style_sort_key(style: str) -> tuple[int, str]:
    return (0 if style.lower() in _REGULAR_STYLES else 1, style.lower())


def group_fonts(
    fonts: tuple[tuple[str, str, str], ...] | None = None,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Group ``(family, style, path)`` rows into one entry per family.

    Each family is ``(name, ((style, path), ...))``. Regular comes first.
    The first file wins when a family has two cuts with the same style name.
    """
    entries = discover_fonts() if fonts is None else fonts
    buckets: dict[str, dict[str, str]] = {}
    for family, style, path in entries:
        name = family or "Default"
        cut = style or "Regular"
        buckets.setdefault(name, {}).setdefault(cut, path)
    grouped = []
    for name in sorted(buckets, key=str.lower):
        styles = tuple(sorted(buckets[name].items(), key=lambda item: _style_sort_key(item[0])))
        grouped.append((name, styles))
    return tuple(grouped)


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


_TEXT_ALIGNS = ("left", "center", "right", "justify")


def render_text(info: TextInfo) -> Image.Image:
    """Rasterize ``info`` to a tight RGBA image.

    Rotation is clockwise, matching the rest of the editor. Pillow itself
    rotates counter-clockwise, so the angle is negated after the glyphs are drawn.

    Alignment only moves lines that are shorter than the longest line. One
    line is already that wide, so left, center, right, and justify look the
    same. Justify spreads the gaps on every line except the last, which stays
    left. A line with a single word cannot grow.
    """
    font = load_font(info.font_path, info.size)
    text = info.text if info.text else " "
    spacing = max(0, int(info.line_spacing))
    align = info.align if info.align in _TEXT_ALIGNS else "left"
    fill = info.color if len(info.color) == 4 else (*info.color[:3], 255)
    if align == "justify" and "\n" in text:
        image = _draw_justified(text, font, fill, spacing)
    else:
        image = _draw_aligned(text, font, fill, spacing, "left" if align == "justify" else align)
    if abs(info.rotation) > 0.01:
        image = image.rotate(-float(info.rotation), resample=Image.Resampling.BICUBIC, expand=True)
    return image


def _draw_aligned(text: str, font, fill, spacing: int, align: str) -> Image.Image:
    probe = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align)
    width = max(1, int(right - left) + 8)
    height = max(1, int(bottom - top) + 8)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.multiline_text((-left + 4, -top + 4), text, font=font, fill=fill, spacing=spacing, align=align)
    return image


def _draw_justified(text: str, font, fill, spacing: int) -> Image.Image:
    """Draw every line except the last out to the longest line's width.

    Spaces are removed and the leftover width is shared across the gaps, so
    the line meets both edges. The last line, a single word, and a one-line
    string stay left. The bitmap is then cropped to the ink plus the same
    4px pad the other alignments use.
    """
    lines = text.split("\n")
    widths = [float(font.getlength(line)) for line in lines]
    target = max(widths) if widths else 0.0
    step = float(font.getbbox("A")[3]) + float(spacing)
    pad = 4
    width = max(1, int(math.ceil(target)) + pad * 2 + 4)
    height = max(1, int(math.ceil(step * max(1, len(lines)))) + int(math.ceil(step)) + pad * 2)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    top = float(pad)
    last = len(lines) - 1
    for index, line in enumerate(lines):
        words = line.split(" ")
        short = target - widths[index] > 0.5
        if index != last and len(words) > 1 and short:
            word_widths = [float(font.getlength(word)) for word in words]
            gap = (target - sum(word_widths)) / (len(words) - 1)
            left = float(pad)
            for word, word_width in zip(words, word_widths):
                if word:
                    draw.text((left, top), word, font=font, fill=fill)
                left += word_width + gap
        else:
            draw.text((pad, top), line, font=font, fill=fill)
        top += step
    return _pad_content(image, pad)


def _pad_content(image: Image.Image, pad: int) -> Image.Image:
    box = image.getbbox()
    if box is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    cropped = image.crop(box)
    out = Image.new("RGBA", (cropped.width + pad * 2, cropped.height + pad * 2), (0, 0, 0, 0))
    out.paste(cropped, (pad, pad))
    return out
