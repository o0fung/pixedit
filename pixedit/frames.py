"""Play and export the frames a document already has.

A document with more than one page plays those pages. A single page whose
pictures are hidden except one plays those layers: that is how a GIF or
multi-frame TIFF opens. A normal stack of visible layers is not an animation.
"""

from __future__ import annotations

from PIL import Image

from pixedit.composite import composite, composite_group
from pixedit.model import Document


def frame_kind(document: Document) -> str | None:
    """``pages``, ``layers``, or None when there is nothing to play."""
    if len(document.groups) > 1:
        return "pages"
    pictures = [layer for layer in document.layers if layer.kind != "redaction"]
    if len(pictures) > 1 and sum(1 for layer in pictures if layer.visible) <= 1:
        return "layers"
    return None


def frame_count(document: Document) -> int:
    kind = frame_kind(document)
    if kind == "pages":
        return len(document.groups)
    if kind == "layers":
        return sum(1 for layer in document.layers if layer.kind != "redaction")
    return 0


def apply_layer_frame(document: Document, index: int) -> None:
    """Show one picture layer and the masks that belong to it.

    Other pictures are hidden. The caller restores the previous visibility.
    This does not record an undo step.
    """
    pictures = [layer for layer in document.layers if layer.kind != "redaction"]
    if not pictures:
        return
    target = pictures[int(index) % len(pictures)]
    for layer in document.layers:
        if layer.kind == "redaction":
            layer.visible = layer.target_id == target.id
        else:
            layer.visible = layer is target


def render_frames(document: Document) -> list[Image.Image]:
    """One RGBA image per frame, at that frame's own size.

    Layer playback hides pictures while it composites, then puts visibility
    back. Page playback does not change which page is on screen.
    """
    kind = frame_kind(document)
    if kind == "pages":
        return [composite_group(group) for group in document.groups]
    if kind != "layers":
        return []
    pictures = [layer for layer in document.layers if layer.kind != "redaction"]
    saved = [(layer, layer.visible) for layer in document.layers]
    frames: list[Image.Image] = []
    try:
        for index in range(len(pictures)):
            apply_layer_frame(document, index)
            frames.append(composite(document).copy())
    finally:
        for layer, visible in saved:
            layer.visible = visible
    return frames


def frame_labels(document: Document) -> list[str]:
    kind = frame_kind(document)
    if kind == "pages":
        return [group.name for group in document.groups]
    if kind == "layers":
        return [layer.name for layer in document.layers if layer.kind != "redaction"]
    return []


def pad_frames(frames: list[Image.Image], labels: list[str]) -> tuple[list[Image.Image], list[str]]:
    """Center smaller frames on the largest size.

    The notes name each frame that was padded, including its old size, so
    the status bar can say which ones changed.
    """
    if not frames:
        return [], []
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    padded: list[Image.Image] = []
    notes: list[str] = []
    for label, frame in zip(labels, frames):
        if frame.size == (width, height):
            padded.append(frame)
            continue
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        canvas.paste(frame.convert("RGBA"), ((width - frame.width) // 2, (height - frame.height) // 2))
        padded.append(canvas)
        notes.append(f"{label} ({frame.width} × {frame.height})")
    return padded, notes


def export_animated_gif(document: Document, path: str, delay_ms: int) -> list[str]:
    """Write one GIF frame per page or hidden frame layer.

    Frames that differ in size are padded to the largest size and named in
    the returned notes. A document that is not an animation raises ValueError.
    GIF has no real alpha, so each frame is composited on white.
    """
    if frame_kind(document) is None:
        raise ValueError(
            "Nothing to export. Open a multi-page file, or a GIF whose frames are hidden layers."
        )
    frames, notes = pad_frames(render_frames(document), frame_labels(document))
    if not frames:
        raise ValueError("Nothing to export.")
    delay = max(20, int(delay_ms))
    sheets: list[Image.Image] = []
    for frame in frames:
        white = Image.new("RGBA", frame.size, (255, 255, 255, 255))
        flat = Image.alpha_composite(white, frame.convert("RGBA"))
        sheets.append(flat.convert("P", palette=Image.Palette.ADAPTIVE))
    first, *rest = sheets
    first.save(
        path,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=delay,
        loop=0,
        disposal=2,
    )
    return notes


def shared_frame_zoom(
    view_w: float,
    view_h: float,
    sizes: list[tuple[int, int]],
    *,
    ruler: int = 0,
    margin: float = 36.0,
) -> float:
    """One view scale for every page, chosen so the largest page fits.

    Smaller pages use the same scale and are letterboxed. The document
    pixels are not resampled.
    """
    if not sizes:
        return 1.0
    max_w = max(1, max(width for width, _height in sizes))
    max_h = max(1, max(height for _width, height in sizes))
    avail_w = max(1.0, float(view_w) - int(ruler) - float(margin))
    avail_h = max(1.0, float(view_h) - int(ruler) - float(margin))
    return min(32.0, max(0.02, min(avail_w / max_w, avail_h / max_h)))


def letterbox_offset(
    view_w: float,
    view_h: float,
    page_w: int,
    page_h: int,
    zoom: float,
    *,
    ruler: int = 0,
) -> tuple[float, float]:
    """Top-left of a page centered in the view at ``zoom``."""
    return (
        (float(view_w) - int(ruler) - int(page_w) * float(zoom)) / 2.0,
        (float(view_h) - int(ruler) - int(page_h) * float(zoom)) / 2.0,
    )

