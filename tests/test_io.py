"""Raster, multi-frame, PDF, and .pix project IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pixedit.composite import composite
from pixedit.factory import redaction_layer
from pixedit.io import export_image, load_path, load_project, load_raster, save_project
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
    assert loaded.selection is not None and loaded.selection.getpixel((1, 1)) == 255
    assert np.array_equal(np.array(composite(doc)), np.array(composite(loaded)))


def test_pdf_pages_become_layers(tmp_path: Path):
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
    assert len(doc.layers) == 2
    assert doc.layers[0].name == "Page 1"
    assert doc.layers[0].visible is True
    assert doc.layers[1].visible is False
    assert doc.layers[0].image.width >= 72


def test_export_flattened(tmp_path: Path):
    session = Session(new_document(8, 6, "white"))
    for suffix in (".png", ".jpg", ".webp"):
        path = tmp_path / f"out{suffix}"
        export_image(session.document, str(path))
        with Image.open(path) as saved:
            assert saved.size == (8, 6)


def test_old_text_project_defaults_align_and_spacing():
    from pixedit.io import _text_from_json

    info = _text_from_json(
        {"text": "A", "font_path": None, "font_family": "Default", "size": 18, "color": [0, 0, 0, 255]}
    )
    assert info is not None
    assert info.align == "left"
    assert info.line_spacing == 4
