"""Open images, multi-page PDFs, and ``.pix`` projects. Export flattened files."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image

from pixedit.composite import composite, composite_over
from pixedit.factory import raster_layer, redaction_layer, text_layer
from pixedit.model import Document, Layer, RedactionInfo, TextInfo

RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
PROJECT_SUFFIX = ".pix"
FORMAT_VERSION = 1


def load_raster(path: str) -> Document:
    """Load a still image, or each frame of a GIF/TIFF, as layers.

    Extra frames are hidden so a multi-page file opens on its first page
    instead of stacking every frame on top of the others.
    """
    with Image.open(path) as source:
        frames: list[Image.Image] = []
        count = getattr(source, "n_frames", 1) or 1
        for index in range(count):
            source.seek(index)
            frames.append(source.convert("RGBA").copy())
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    stem = Path(path).stem
    layers: list[Layer] = []
    multiple = len(frames) > 1
    for index, frame in enumerate(frames):
        name = f"{stem} {index + 1}" if multiple else stem
        layer = raster_layer(name, frame)
        layer.visible = index == 0 or not multiple
        layers.append(layer)
    return Document(width=width, height=height, layers=layers)


def load_pdf(path: str, scale: float = 2.0) -> Document:
    """Render each PDF page to its own layer.

    ``scale`` is the PDFium zoom. 1.0 is about 72 DPI and 2.0 is about 144 DPI.
    Only the first page is visible; the rest stay in the layer stack.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    pages: list[Image.Image] = []
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            bitmap = page.render(scale=float(scale))
            try:
                pages.append(bitmap.to_pil().convert("RGBA"))
            finally:
                bitmap.close()
                page.close()
    finally:
        pdf.close()
    if not pages:
        raise ValueError("PDF has no pages")
    width = max(page.width for page in pages)
    height = max(page.height for page in pages)
    layers = []
    for index, page in enumerate(pages):
        layer = raster_layer(f"Page {index + 1}", page)
        layer.visible = index == 0
        layers.append(layer)
    return Document(width=width, height=height, layers=layers)


def _text_from_json(payload: dict | None) -> TextInfo | None:
    if not payload:
        return None
    color = tuple(int(c) for c in payload["color"])
    if len(color) == 3:
        color = (*color, 255)
    return TextInfo(
        text=str(payload.get("text", "")),
        font_path=payload.get("font_path"),
        font_family=str(payload.get("font_family", "")),
        size=int(payload.get("size", 32)),
        color=(color[0], color[1], color[2], color[3]),
        rotation=float(payload.get("rotation", 0)),
        align=str(payload.get("align", "left") or "left"),
        line_spacing=max(0, int(payload.get("line_spacing", 4))),
    )


def _text_to_json(info: TextInfo | None) -> dict | None:
    if info is None:
        return None
    return {
        "text": info.text,
        "font_path": info.font_path,
        "font_family": info.font_family,
        "size": info.size,
        "color": list(info.color),
        "rotation": info.rotation,
        "align": info.align,
        "line_spacing": info.line_spacing,
    }


def save_project(document: Document, path: str) -> None:
    """Write a ``.pix`` zip: ``manifest.json`` plus PNG assets for each layer."""
    destination = Path(path)
    manifest_layers = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, layer in enumerate(document.layers):
            image_name = f"layers/{index:03d}.png"
            mask_name = None
            with archive.open(image_name, "w") as handle:
                layer.image.save(handle, format="PNG")
            if layer.mask is not None:
                mask_name = f"layers/{index:03d}_mask.png"
                with archive.open(mask_name, "w") as handle:
                    layer.mask.save(handle, format="PNG")
            redaction = None
            if layer.redaction is not None:
                redaction = {
                    "kind": layer.redaction.kind,
                    "amount": layer.redaction.amount,
                    "color": list(layer.redaction.color),
                    "seed": int(layer.redaction.seed),
                }
            manifest_layers.append(
                {
                    "id": layer.id,
                    "name": layer.name,
                    "kind": layer.kind,
                    "visible": layer.visible,
                    "opacity": layer.opacity,
                    "blend_mode": layer.blend_mode,
                    "x": layer.x,
                    "y": layer.y,
                    "image": image_name,
                    "mask": mask_name,
                    "text": _text_to_json(layer.text),
                    "redaction": redaction,
                }
            )
        if document.selection is not None:
            with archive.open("selection.png", "w") as handle:
                document.selection.save(handle, format="PNG")
        manifest = {
            "format": "pixedit",
            "version": FORMAT_VERSION,
            "width": document.width,
            "height": document.height,
            "selection": "selection.png" if document.selection is not None else None,
            "layers": manifest_layers,
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))


def load_project(path: str) -> Document:
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != "pixedit":
            raise ValueError("Not a PixEdit project")
        layers: list[Layer] = []
        for entry in manifest["layers"]:
            with archive.open(entry["image"]) as handle:
                image = Image.open(handle).convert("RGBA")
                image.load()
            mask = None
            if entry.get("mask"):
                with archive.open(entry["mask"]) as handle:
                    mask = Image.open(handle).convert("L")
                    mask.load()
            text = _text_from_json(entry.get("text"))
            redaction_payload = entry.get("redaction")
            redaction = None
            if redaction_payload:
                raw_color = redaction_payload.get("color", [0, 0, 0, 255])
                if not isinstance(raw_color, (list, tuple)) or len(raw_color) < 3:
                    raw_color = [0, 0, 0, 255]
                if len(raw_color) == 3:
                    raw_color = [*raw_color, 255]
                redaction = RedactionInfo(
                    kind=str(redaction_payload.get("kind", "pixelate")),
                    amount=float(redaction_payload.get("amount", 12)),
                    color=tuple(int(channel) for channel in raw_color[:4]),
                    seed=int(redaction_payload.get("seed", 1)),
                )
            kind = entry.get("kind", "raster")
            if kind == "text" and text is not None:
                layer = text_layer(entry.get("name", "Text"), text, int(entry.get("x", 0)), int(entry.get("y", 0)))
                # Prefer the stored render so a missing font does not move pixels.
                layer.image = image
                layer.id = entry["id"]
            elif kind == "redaction":
                info = redaction or RedactionInfo()
                layer = redaction_layer(
                    entry.get("name", "Mask"),
                    mask or Image.new("L", (manifest["width"], manifest["height"]), 0),
                    info.kind,
                    info.amount,
                    int(entry.get("x", 0)),
                    int(entry.get("y", 0)),
                    info.color,
                    info.seed,
                )
                layer.id = entry["id"]
            else:
                layer = raster_layer(entry.get("name", "Layer"), image, int(entry.get("x", 0)), int(entry.get("y", 0)))
                layer.id = entry["id"]
            layer.visible = bool(entry.get("visible", True))
            layer.opacity = float(entry.get("opacity", 1))
            layer.blend_mode = str(entry.get("blend_mode", "normal"))
            if mask is not None:
                layer.mask = mask
            layers.append(layer)
        selection = None
        if manifest.get("selection"):
            with archive.open(manifest["selection"]) as handle:
                selection = Image.open(handle).convert("L")
                selection.load()
    return Document(
        width=int(manifest["width"]),
        height=int(manifest["height"]),
        layers=layers,
        selection=selection,
    )


def load_path(path: str, pdf_scale: float = 2.0) -> Document:
    suffix = Path(path).suffix.lower()
    if suffix == PROJECT_SUFFIX:
        return load_project(path)
    if suffix == ".pdf":
        return load_pdf(path, scale=pdf_scale)
    if suffix in RASTER_SUFFIXES or suffix == "":
        return load_raster(path)
    # Try Pillow first, then PDFium, so odd extensions still open when they can.
    try:
        return load_raster(path)
    except Exception:
        return load_pdf(path, scale=pdf_scale)


def export_image(document: Document, path: str, fmt: str | None = None, quality: int = 92) -> None:
    """Write a flattened PNG, JPEG, or WebP file."""
    suffix = (fmt or Path(path).suffix.lstrip(".")).lower()
    if suffix in {"jpg", "jpeg"}:
        flat = composite_over(document, (255, 255, 255, 255)).convert("RGB")
        flat.save(path, format="JPEG", quality=quality)
        return
    flat = composite(document)
    if suffix == "webp":
        flat.save(path, format="WEBP", quality=quality)
        return
    flat.save(path, format="PNG")
