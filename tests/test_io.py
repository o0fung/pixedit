"""Raster, multi-frame, PDF, and .pix project IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixedit.composite import composite
from pixedit.factory import redaction_layer
from pixedit.io import (
    export_filter_for_suffix,
    export_filter_string,
    export_image,
    extension_for_export_filter,
    load_path,
    load_project,
    load_raster,
    save_project,
)
from pixedit.model import Document, TextInfo, new_document
from pixedit.session import Session
from pixedit.textutil import render_text


def test_roundtrip_common_rasters_and_gif_frames(tmp_path: Path):
    image = Image.new("RGBA", (6, 4), (10, 20, 30, 255))
    image.putpixel((1, 1), (9, 8, 7, 255))
    for suffix, mode in ((".png", "RGBA"), (".jpg", "RGB"), (".webp", "RGBA"), (".tif", "RGBA"), (".bmp", "RGB")):
        path = tmp_path / f"sample{suffix}"
        saved = image.convert(mode) if mode == "RGB" else image
        # WebP is lossy unless asked otherwise; the other formats here are lossless.
        if suffix == ".webp":
            saved.save(path, lossless=True)
        else:
            saved.save(path)
        doc = load_raster(str(path))
        assert doc.width >= 1 and len(doc.layers) == 1
        if suffix != ".jpg":
            assert doc.layers[0].image.getpixel((1, 1))[:3] == (9, 8, 7)

    frames = [Image.new("RGBA", (5, 5), (255, 0, 0, 255)), Image.new("RGBA", (5, 5), (0, 0, 255, 255))]
    gif = tmp_path / "anim.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=80, disposal=2)
    animated = load_raster(str(gif))
    assert len(animated.layers) == 2
    assert animated.layers[0].visible is True
    assert animated.layers[1].visible is False


def test_project_roundtrip_keeps_text_redaction_and_selection(tmp_path: Path):
    info = TextInfo("A", None, "Default", 18, (0, 0, 0, 255), 12, "center", 16)
    text_image = render_text(info)
    from pixedit.factory import text_layer

    layer = text_layer("Title", info, 3, 4)
    layer.image = text_image
    mask = Image.new("L", (30, 20), 0)
    mask.paste(255, (5, 5, 12, 14))
    redact = redaction_layer("Privacy", mask, "pixelate", 9)
    base = Image.new("RGBA", (30, 20), (240, 240, 240, 255))
    from pixedit.factory import raster_layer

    doc = Document(30, 20, [raster_layer("Base", base), layer, redact])
    doc.selection = Image.new("L", (30, 20), 0)
    doc.selection.paste(255, (0, 0, 4, 4))
    path = tmp_path / "page.pix"
    save_project(doc, str(path))
    loaded = load_project(str(path))
    assert (loaded.width, loaded.height) == (30, 20)
    assert [item.name for item in loaded.layers] == ["Base", "Title", "Privacy"]
    assert loaded.layers[1].kind == "text"
    assert loaded.layers[1].text is not None
    assert loaded.layers[1].text.rotation == 12
    assert loaded.layers[1].text.align == "center"
    assert loaded.layers[1].text.line_spacing == 16
    assert loaded.layers[1].x == 3
    assert loaded.layers[2].kind == "redaction"
    assert loaded.layers[2].redaction is not None
    assert loaded.layers[2].redaction.amount == 9
    assert loaded.layers[2].redaction.seed == redact.redaction.seed
    assert loaded.layers[2].redaction.color == redact.redaction.color
    assert loaded.layers[2].mask.getpixel((6, 6)) == 255
    assert loaded.layers[2].target_id == loaded.layers[1].id
    assert loaded.selection is not None and loaded.selection.getpixel((1, 1)) == 255
    assert np.array_equal(np.array(composite(doc)), np.array(composite(loaded)))


def test_pdf_pages_become_groups(tmp_path: Path):
    pdfium = __import__("pypdfium2")
    pdf_path = tmp_path / "pages.pdf"
    pdf = pdfium.PdfDocument.new()
    try:
        pdf.new_page(72, 72)
        pdf.new_page(72, 36)
        pdf.save(str(pdf_path))
    finally:
        pdf.close()
    doc = load_path(str(pdf_path), pdf_scale=1)
    assert [group.name for group in doc.groups] == ["Page 1", "Page 2"]
    assert doc.active_group_id == doc.groups[0].id
    assert all(len(group.layers) == 1 and group.layers[0].visible for group in doc.groups)
    assert doc.groups[0].layers[0].name == "Page 1"
    assert doc.groups[0].width >= 72
    assert doc.groups[0].height >= 72
    assert doc.groups[1].width >= 72
    assert doc.groups[1].height < doc.groups[0].height
    assert doc.width == doc.groups[0].width
    assert doc.height == doc.groups[0].height
    assert doc.dpi == 72
    sharp = load_path(str(pdf_path), pdf_scale=2)
    assert sharp.dpi == 144
    assert sharp.groups[0].width > doc.groups[0].width


def test_export_flattened(tmp_path: Path):
    session = Session(new_document(8, 6, "white"))
    for suffix in (".png", ".jpg", ".webp", ".tif", ".tiff", ".bmp", ".gif"):
        path = tmp_path / f"out{suffix}"
        export_image(session.document, str(path))
        with Image.open(path) as saved:
            assert saved.size == (8, 6)


def test_export_keeps_alpha_or_flattens_on_white(tmp_path: Path):
    doc = new_document(4, 4, "transparent")
    for suffix in (".png", ".tif"):
        path = tmp_path / f"clear{suffix}"
        export_image(doc, str(path))
        with Image.open(path) as saved:
            assert saved.convert("RGBA").getpixel((0, 0))[3] == 0
    for suffix in (".jpg", ".bmp", ".gif"):
        path = tmp_path / f"white{suffix}"
        export_image(doc, str(path))
        with Image.open(path) as saved:
            assert saved.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_export_pdf_is_one_page_at_document_dpi(tmp_path: Path):
    doc = new_document(144, 72, "transparent")
    doc.dpi = 144
    path = tmp_path / "out.pdf"
    export_image(doc, str(path))
    pdfium = __import__("pypdfium2")
    pdf = pdfium.PdfDocument(str(path))
    try:
        assert len(pdf) == 1
        # 144 px at 144 dpi is 72 points. 72 px at 144 dpi is 36 points.
        width, height = pdf[0].get_size()
        assert abs(width - 72) < 1
        assert abs(height - 36) < 1
        bitmap = pdf[0].render(scale=1)
        try:
            image = bitmap.to_pil().convert("RGB")
            assert image.getpixel((0, 0)) == (255, 255, 255)
        finally:
            bitmap.close()
    finally:
        pdf.close()


def test_export_pdf_keeps_group_order_and_isolates_edits(tmp_path: Path):
    pdfium = __import__("pypdfium2")
    source = tmp_path / "pages.pdf"
    pdf = pdfium.PdfDocument.new()
    try:
        pdf.new_page(72, 72)
        pdf.new_page(72, 36)
        pdf.save(str(source))
    finally:
        pdf.close()
    doc = load_path(str(source), pdf_scale=1)
    page_two = doc.groups[1].layers[0]
    page_two.image.paste((255, 0, 0, 255), (0, 0, page_two.image.width, page_two.image.height))
    untouched = doc.groups[0].layers[0].image.getpixel((0, 0))
    assert doc.active_group_id == doc.groups[0].id
    path = tmp_path / "out.pdf"
    export_image(doc, str(path))
    assert doc.active_group_id == doc.groups[0].id
    assert doc.groups[0].layers[0].image.getpixel((0, 0)) == untouched
    opened = pdfium.PdfDocument(str(path))
    try:
        assert len(opened) == 2
        first_w, first_h = opened[0].get_size()
        second_w, second_h = opened[1].get_size()
        assert abs(first_w - doc.groups[0].width) < 1
        assert abs(first_h - doc.groups[0].height) < 1
        assert abs(second_w - doc.groups[1].width) < 1
        assert abs(second_h - doc.groups[1].height) < 1
        assert second_h < first_h
        first = opened[0].render(scale=1)
        second = opened[1].render(scale=1)
        try:
            top_page = first.to_pil().convert("RGB")
            edited = second.to_pil().convert("RGB")
            assert top_page.getpixel((0, 0)) != (255, 0, 0)
            red, green, blue = edited.getpixel((edited.width // 2, edited.height // 2))
            assert red > 200 and green < 40 and blue < 40
        finally:
            first.close()
            second.close()
    finally:
        opened.close()


def test_single_image_export_uses_the_page_on_screen(tmp_path: Path):
    doc = new_document(8, 6, "white")
    session = Session(doc)
    session.add_group()
    second = session.document.active_group_id
    session.document.width = 5
    session.document.height = 4
    session.activate_group(session.document.groups[0].id)
    wide = tmp_path / "wide.png"
    export_image(doc, str(wide))
    with Image.open(wide) as saved:
        assert saved.size == (8, 6)
    session.activate_group(second)
    narrow = tmp_path / "narrow.png"
    export_image(doc, str(narrow))
    with Image.open(narrow) as saved:
        assert saved.size == (5, 4)


def test_v1_project_opens_as_one_group_and_saves_as_v2(tmp_path: Path):
    import json
    import zipfile
    from io import BytesIO

    image = Image.new("RGBA", (4, 4), (8, 9, 10, 255))
    image.putpixel((1, 1), (3, 4, 5, 255))
    selection = Image.new("L", (4, 4), 0)
    selection.putpixel((2, 2), 255)
    payload = BytesIO()
    image.save(payload, format="PNG")
    mask_payload = BytesIO()
    selection.save(mask_payload, format="PNG")
    manifest = {
        "format": "pixedit",
        "version": 1,
        "width": 4,
        "height": 4,
        "dpi": 144,
        "selection": "selection.png",
        "layers": [
            {
                "id": "photo",
                "name": "Photo",
                "kind": "raster",
                "visible": True,
                "opacity": 1,
                "blend_mode": "normal",
                "x": 0,
                "y": 0,
                "image": "layers/000.png",
                "mask": None,
                "text": None,
                "redaction": None,
                "target_id": None,
            }
        ],
    }
    legacy = tmp_path / "legacy.pix"
    with zipfile.ZipFile(legacy, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("layers/000.png", payload.getvalue())
        archive.writestr("selection.png", mask_payload.getvalue())
    loaded = load_project(str(legacy))
    assert len(loaded.groups) == 1
    assert loaded.groups[0].name == "Group 1"
    assert loaded.width == 4 and loaded.height == 4
    assert loaded.dpi == 144
    assert loaded.layers[0].name == "Photo"
    assert loaded.layers[0].image.getpixel((1, 1)) == (3, 4, 5, 255)
    assert loaded.selection is not None
    assert loaded.selection.getpixel((2, 2)) == 255

    saved = tmp_path / "current.pix"
    save_project(loaded, str(saved))
    with zipfile.ZipFile(saved) as archive:
        written = json.loads(archive.read("manifest.json"))
    assert written["version"] == 2
    assert written["groups"][0]["name"] == "Group 1"
    assert written["groups"][0]["layers"][0]["name"] == "Photo"
    again = load_project(str(saved))
    assert again.layers[0].image.getpixel((1, 1)) == (3, 4, 5, 255)
    assert again.selection is not None and again.selection.getpixel((2, 2)) == 255

    written["version"] = 9
    future = tmp_path / "future.pix"
    with zipfile.ZipFile(saved) as archive:
        blobs = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(future, "w") as archive:
        for name, blob in blobs.items():
            if name == "manifest.json":
                archive.writestr(name, json.dumps(written))
            else:
                archive.writestr(name, blob)
    with pytest.raises(ValueError, match="version 9"):
        load_project(str(future))


def test_project_roundtrip_keeps_two_groups(tmp_path: Path):
    session = Session(new_document(6, 4, "white"))
    session.document.groups[0].name = "Page 1"
    session.document.layers[0].image.putpixel((0, 0), (1, 2, 3, 255))
    session.add_group()
    session.rename_group(session.document.active_group_id, "Page 2")
    session.document.width = 9
    session.document.height = 3
    session.document.layers[0].image.putpixel((1, 1), (9, 8, 7, 255))
    path = tmp_path / "pages.pix"
    save_project(session.document, str(path))
    loaded = load_project(str(path))
    assert [group.name for group in loaded.groups] == ["Page 1", "Page 2"]
    assert loaded.group.name == "Page 2"
    assert loaded.width == 9 and loaded.height == 3
    assert loaded.groups[0].width == 6 and loaded.groups[0].height == 4
    assert loaded.groups[0].layers[0].image.getpixel((0, 0)) == (1, 2, 3, 255)
    assert loaded.layers[0].image.getpixel((1, 1)) == (9, 8, 7, 255)


def test_export_filter_lists_every_written_format():
    text = export_filter_string()
    for label in ("PNG", "JPEG", "WebP", "TIFF", "BMP", "GIF", "PDF"):
        assert label in text
    assert export_filter_for_suffix(".pdf").startswith("PDF")
    assert export_filter_for_suffix(".jpeg").startswith("JPEG")
    assert export_filter_for_suffix(".pix").startswith("PNG")
    assert extension_for_export_filter("PDF (*.pdf)") == ".pdf"
    assert extension_for_export_filter("JPEG (*.jpg *.jpeg)") == ".jpg"
    assert extension_for_export_filter("TIFF (*.tif *.tiff)") == ".tif"
    assert extension_for_export_filter("") == ".png"


def test_export_rejects_unknown_suffix(tmp_path: Path):
    with pytest.raises(ValueError):
        export_image(new_document(2, 2, "white"), str(tmp_path / "out.xyz"))


def test_dpi_roundtrip_and_missing_project_key(tmp_path: Path):
    image = Image.new("RGB", (4, 4), (1, 2, 3))
    raster_path = tmp_path / "dpi.png"
    image.save(raster_path, dpi=(180, 180))
    loaded = load_raster(str(raster_path))
    assert abs(loaded.dpi - 180) < 1

    doc = new_document(4, 4, "white")
    doc.dpi = 180
    exported = tmp_path / "out.png"
    export_image(doc, str(exported))
    assert abs(load_raster(str(exported)).dpi - 180) < 1

    doc.dpi = 300
    project = tmp_path / "dpi.pix"
    save_project(doc, str(project))
    assert load_project(str(project)).dpi == 300

    import json
    import zipfile

    with zipfile.ZipFile(project) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        blobs = {name: archive.read(name) for name in archive.namelist()}
    manifest.pop("dpi", None)
    bare = tmp_path / "bare.pix"
    with zipfile.ZipFile(bare, "w") as archive:
        for name, blob in blobs.items():
            if name == "manifest.json":
                archive.writestr(name, json.dumps(manifest))
            else:
                archive.writestr(name, blob)
    assert load_project(str(bare)).dpi == 72


def test_old_text_project_defaults_align_and_spacing():
    from pixedit.io import _text_from_json

    info = _text_from_json(
        {"text": "A", "font_path": None, "font_family": "Default", "size": 18, "color": [0, 0, 0, 255]}
    )
    assert info is not None
    assert info.align == "left"
    assert info.line_spacing == 4
