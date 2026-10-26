"""Open images, multi-page PDFs, and ``.pix`` projects. Export flattened files."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image

from pixedit.composite import composite, composite_group_over, composite_over
from pixedit.factory import raster_layer, redaction_layer, text_layer
from pixedit.model import Document, Layer, LayerGroup, RedactionInfo, TextInfo, bind_mask_targets, new_id

RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
PROJECT_SUFFIX = ".pix"
FORMAT_VERSION = 2
DEFAULT_DPI = 72.0

# Export dialog order. The first suffix is appended when the chosen name has none.
# JPEG, BMP, GIF, and PDF are flattened on white inside export_image.
EXPORT_CHOICES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PNG", (".png",)),
    ("JPEG", (".jpg", ".jpeg")),
    ("WebP", (".webp",)),
    ("TIFF", (".tif", ".tiff")),
    ("BMP", (".bmp",)),
    ("GIF", (".gif",)),
    ("PDF", (".pdf",)),
)
EXPORT_SUFFIXES = {suffix for _label, suffixes in EXPORT_CHOICES for suffix in suffixes}
# No straight alpha. Clear pixels would turn black if the channel were dropped.
_OPAQUE_EXPORT_SUFFIXES = {"jpg", "jpeg", "bmp", "gif", "pdf"}


def _export_filter(label: str, suffixes: tuple[str, ...]) -> str:
    globs = " ".join(f"*{suffix}" for suffix in suffixes)
    return f"{label} ({globs})"


def export_filter_string() -> str:
    """QFileDialog filter listing every format ``export_image`` can write."""
    return ";;".join(_export_filter(label, suffixes) for label, suffixes in EXPORT_CHOICES)


def export_filter_for_suffix(suffix: str) -> str:
    """The Export filter that matches ``suffix``. Unknown suffixes use PNG."""
    wanted = suffix.lower()
    for label, suffixes in EXPORT_CHOICES:
        if wanted in suffixes:
            return _export_filter(label, suffixes)
    label, suffixes = EXPORT_CHOICES[0]
    return _export_filter(label, suffixes)


def extension_for_export_filter(selected: str) -> str:
    """Suffix to add when the chosen export name has no known extension."""
    for label, suffixes in EXPORT_CHOICES:
        if selected.startswith(label):
            return suffixes[0]
    return ".png"


def image_dpi(image: Image.Image) -> float:
    """Resolution stored on a file, or 72 when the file does not say."""
    raw = image.info.get("dpi")
    if isinstance(raw, (tuple, list)) and raw:
        try:
            value = float(raw[0])
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    density = image.info.get("jfif_density")
    unit = image.info.get("jfif_unit")
    if isinstance(density, (tuple, list)) and density:
        try:
            value = float(density[0])
        except (TypeError, ValueError):
            value = 0.0
        if value > 0 and unit == 1:
            return value
        if value > 0 and unit == 2:
            return value * 2.54
    return DEFAULT_DPI


def load_raster(path: str) -> Document:
    """Load a still image, or each frame of a GIF/TIFF, as layers.

    Extra frames are hidden so a multi-page file opens on its first page
    instead of stacking every frame on top of the others.
    """
    with Image.open(path) as source:
        dpi = image_dpi(source)
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
    return Document(width=width, height=height, layers=layers, dpi=dpi)


def load_pdf(path: str, scale: float = 2.0) -> Document:
    """Render each PDF page as its own group.

    ``scale`` is the PDFium zoom. 1.0 is about 72 DPI and 2.0 is about 144 DPI.
    Every page stays visible inside its group. The canvas shows the first page.
    Groups are stored in page order, which is the order PDF export writes.
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
    groups: list[LayerGroup] = []
    for index, page in enumerate(pages):
        layer = raster_layer(f"Page {index + 1}", page)
        groups.append(
            LayerGroup(
                id=new_id(),
                name=f"Page {index + 1}",
                width=page.width,
                height=page.height,
                layers=[layer],
            )
        )
    return Document(
        width=groups[0].width,
        height=groups[0].height,
        dpi=DEFAULT_DPI * float(scale),
        groups=groups,
        active_group_id=groups[0].id,
    )


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


def _dpi_from_manifest(manifest: dict) -> float:
    dpi = DEFAULT_DPI
    if manifest.get("dpi") is not None:
        try:
            dpi = float(manifest["dpi"])
        except (TypeError, ValueError):
            dpi = DEFAULT_DPI
        if dpi <= 0:
            dpi = DEFAULT_DPI
    return dpi


def _layer_manifest(layer: Layer, image_name: str, mask_name: str | None) -> dict:
    redaction = None
    if layer.redaction is not None:
        redaction = {
            "kind": layer.redaction.kind,
            "amount": layer.redaction.amount,
            "color": list(layer.redaction.color),
            "seed": int(layer.redaction.seed),
        }
    return {
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
        "target_id": layer.target_id,
    }


def _read_layer(archive: zipfile.ZipFile, entry: dict, width: int, height: int) -> Layer:
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
            mask or Image.new("L", (width, height), 0),
            info.kind,
            info.amount,
            int(entry.get("x", 0)),
            int(entry.get("y", 0)),
            info.color,
            info.seed,
        )
        layer.id = entry["id"]
        raw_target = entry.get("target_id")
        if isinstance(raw_target, str) and raw_target:
            layer.target_id = raw_target
    else:
        layer = raster_layer(entry.get("name", "Layer"), image, int(entry.get("x", 0)), int(entry.get("y", 0)))
        layer.id = entry["id"]
    layer.visible = bool(entry.get("visible", True))
    layer.opacity = float(entry.get("opacity", 1))
    layer.blend_mode = str(entry.get("blend_mode", "normal"))
    if mask is not None:
        layer.mask = mask
    return layer


def _guides_from_entry(entry: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Guide positions stored on a page. Missing or junk values are skipped."""
    raw = entry.get("guides")
    if not isinstance(raw, dict):
        return (), ()

    def clean(key: str) -> tuple[int, ...]:
        values = raw.get(key)
        if not isinstance(values, list):
            return ()
        cleaned = []
        for value in values:
            try:
                cleaned.append(int(value))
            except (TypeError, ValueError):
                continue
        return tuple(cleaned)

    return clean("vertical"), clean("horizontal")


def _read_selection(archive: zipfile.ZipFile, name: str | None) -> Image.Image | None:
    if not name:
        return None
    with archive.open(name) as handle:
        selection = Image.open(handle).convert("L")
        selection.load()
    return selection


def save_project(document: Document, path: str) -> None:
    """Write a version 2 ``.pix`` zip: one folder of PNGs per group."""
    destination = Path(path)
    manifest_groups = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for group_index, group in enumerate(document.groups):
            manifest_layers = []
            for index, layer in enumerate(group.layers):
                image_name = f"groups/{group_index:03d}/layers/{index:03d}.png"
                mask_name = None
                with archive.open(image_name, "w") as handle:
                    layer.image.save(handle, format="PNG")
                if layer.mask is not None:
                    mask_name = f"groups/{group_index:03d}/layers/{index:03d}_mask.png"
                    with archive.open(mask_name, "w") as handle:
                        layer.mask.save(handle, format="PNG")
                manifest_layers.append(_layer_manifest(layer, image_name, mask_name))
            selection_name = None
            if group.selection is not None:
                selection_name = f"groups/{group_index:03d}/selection.png"
                with archive.open(selection_name, "w") as handle:
                    group.selection.save(handle, format="PNG")
            manifest_groups.append(
                {
                    "id": group.id,
                    "name": group.name,
                    "width": group.width,
                    "height": group.height,
                    "selection": selection_name,
                    "layers": manifest_layers,
                    "guides": {
                        "vertical": list(group.guides_v),
                        "horizontal": list(group.guides_h),
                    },
                }
            )
        manifest = {
            "format": "pixedit",
            "version": FORMAT_VERSION,
            "dpi": float(document.dpi),
            "active_group": document.active_group_id,
            "groups": manifest_groups,
        }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))


def _load_project_v1(archive: zipfile.ZipFile, manifest: dict) -> Document:
    """Wrap a one-canvas project as a single group named Group 1.

    The file on disk is left unchanged. The next save writes version 2.
    """
    width = int(manifest["width"])
    height = int(manifest["height"])
    layers = [_read_layer(archive, entry, width, height) for entry in manifest["layers"]]
    # Older projects omit target_id. The mask then belongs to the picture
    # or text layer directly under it.
    bind_mask_targets(layers)
    group = LayerGroup(
        id=new_id(),
        name="Group 1",
        width=width,
        height=height,
        layers=layers,
        selection=_read_selection(archive, manifest.get("selection")),
    )
    return Document(
        width=width,
        height=height,
        dpi=_dpi_from_manifest(manifest),
        groups=[group],
        active_group_id=group.id,
    )


def _load_project_v2(archive: zipfile.ZipFile, manifest: dict) -> Document:
    raw_groups = manifest.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("This PixEdit project has no groups")
    groups: list[LayerGroup] = []
    for entry in raw_groups:
        width = int(entry["width"])
        height = int(entry["height"])
        layers = [_read_layer(archive, layer_entry, width, height) for layer_entry in entry.get("layers", [])]
        bind_mask_targets(layers)
        vertical, horizontal = _guides_from_entry(entry)
        groups.append(
            LayerGroup(
                id=str(entry.get("id") or new_id()),
                name=str(entry.get("name") or "Group"),
                width=width,
                height=height,
                layers=layers,
                selection=_read_selection(archive, entry.get("selection")),
                guides_v=vertical,
                guides_h=horizontal,
            )
        )
    active = manifest.get("active_group")
    if not isinstance(active, str):
        active = groups[0].id
    return Document(
        width=groups[0].width,
        height=groups[0].height,
        dpi=_dpi_from_manifest(manifest),
        groups=groups,
        active_group_id=active,
    )


def load_project(path: str) -> Document:
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != "pixedit":
            raise ValueError("Not a PixEdit project")
        raw_version = manifest.get("version", 1)
        try:
            version = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise ValueError("This PixEdit project has an unreadable version.") from exc
        # Version 1 is one canvas. Opening it builds one group. The next Save
        # writes version 2. Any other version is refused.
        if version <= 1:
            return _load_project_v1(archive, manifest)
        if version == FORMAT_VERSION:
            return _load_project_v2(archive, manifest)
        raise ValueError(
            f"This PixEdit project uses version {version}, which this version of PixEdit cannot open."
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
    """Write a flattened file.

    PNG, WebP, and TIFF keep transparency. JPEG, BMP, GIF, and PDF are
    composited on white, because those formats cannot keep a real alpha
    channel. GIF is reduced to 256 colors. PNG and the other single-image
    formats save the page on screen. PDF writes every group, in group order,
    one raster page each, at the document resolution.
    """
    suffix = (fmt or Path(path).suffix.lstrip(".")).lower()
    # Empty means "caller passed a path with no extension"; write PNG.
    # Anything else that we cannot write is refused before compositing.
    if suffix not in {"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif", "pdf", ""}:
        raise ValueError(f"Unsupported export format: .{suffix}")
    dpi = max(1.0, float(document.dpi))
    resolution = (dpi, dpi)
    if suffix == "pdf":
        pages = [composite_group_over(group, (255, 255, 255, 255)).convert("RGB") for group in document.groups]
        if not pages:
            raise ValueError("Nothing to export")
        first, *rest = pages
        if rest:
            first.save(path, format="PDF", resolution=dpi, save_all=True, append_images=rest)
        else:
            first.save(path, format="PDF", resolution=dpi)
        return
    if suffix in _OPAQUE_EXPORT_SUFFIXES:
        flat = composite_over(document, (255, 255, 255, 255)).convert("RGB")
    else:
        flat = composite(document)
    if suffix in {"jpg", "jpeg"}:
        flat.save(path, format="JPEG", quality=quality, dpi=resolution)
        return
    if suffix == "webp":
        flat.save(path, format="WEBP", quality=quality, dpi=resolution)
        return
    if suffix in {"tif", "tiff"}:
        flat.save(path, format="TIFF", dpi=resolution)
        return
    if suffix == "bmp":
        flat.save(path, format="BMP")
        return
    if suffix == "gif":
        flat.convert("P", palette=Image.Palette.ADAPTIVE).save(path, format="GIF")
        return
    flat.save(path, format="PNG", dpi=resolution)
