"""Canvas templates and the New canvas dialog."""

from __future__ import annotations

import pytest

from pixedit.model import new_document
from pixedit.presets import page_label, preset_size
from pixedit.session import Session


def test_paper_and_screen_preset_pixels():
    assert preset_size("a4", "portrait", 300) == (2480, 3508, 300.0)
    assert preset_size("a4", "landscape", 300) == (3508, 2480, 300.0)
    assert preset_size("a4", "portrait", 72) == (595, 842, 72.0)
    assert preset_size("a4", "portrait", 150) == (1240, 1754, 150.0)
    assert preset_size("a3", "portrait", 300) == (3508, 4961, 300.0)
    assert preset_size("a5", "portrait", 300) == (1748, 2480, 300.0)
    assert preset_size("letter", "portrait", 300) == (2550, 3300, 300.0)
    assert preset_size("letter", "landscape", 300) == (3300, 2550, 300.0)
    # Screen sizes ignore orientation and the dpi argument.
    assert preset_size("hd", "landscape", 300) == (1920, 1080, 72.0)
    assert preset_size("uhd", "portrait", 72) == (3840, 2160, 72.0)
    assert preset_size("square", "portrait", 150) == (1080, 1080, 72.0)
    assert preset_size("custom", "portrait", 300) == (1280, 800, 72.0)
    assert preset_size("a4", "portrait", 0) == (595, 842, 72.0)


def test_page_labels():
    assert page_label("a4", "portrait") == "210 × 297 mm"
    assert page_label("a4", "landscape") == "297 × 210 mm"
    assert page_label("letter", "portrait") == "8.5 × 11 in"
    assert page_label("letter", "landscape") == "11 × 8.5 in"
    assert page_label("hd", "portrait") is None
    assert page_label("custom", "portrait") is None
    with pytest.raises(ValueError):
        preset_size("poster", "portrait", 300)
    with pytest.raises(ValueError):
        preset_size("a4", "sideways", 300)


def test_new_document_stores_dpi():
    document = new_document(10, 12, "black", 300)
    assert document.dpi == 300
    assert document.layers[0].image.size == (10, 12)
    assert document.layers[0].image.getpixel((0, 0)) == (0, 0, 0, 255)
    assert new_document(4, 4, "white", 0).dpi == 72
    session = Session()
    session.new_canvas(8, 6, "transparent", 150)
    assert (session.document.width, session.document.height, session.document.dpi) == (8, 6, 150)


def test_new_document_dialog_templates():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.dialogs import NewDocumentDialog

    _app = QApplication.instance() or QApplication([])
    dialog = NewDocumentDialog()
    assert dialog.values() == (1280, 800, "white", 72.0)
    assert not dialog.orientation.isEnabled()
    assert dialog.resolution.isEnabled()
    assert dialog.summary.text() == "1280 × 800 px"

    dialog.resolution.setCurrentIndex(dialog.resolution.findData(150))
    assert dialog.values() == (1280, 800, "white", 150.0)

    dialog.template.setCurrentIndex(dialog.template.findData("a4"))
    assert dialog.values() == (2480, 3508, "white", 300.0)
    assert dialog.orientation.isEnabled()
    assert dialog.summary.text() == "A4 portrait · 210 × 297 mm · 2480 × 3508 px"

    dialog.orientation.setCurrentIndex(dialog.orientation.findData("landscape"))
    assert dialog.values()[:2] == (3508, 2480)
    assert dialog.summary.text() == "A4 landscape · 297 × 210 mm · 3508 × 2480 px"

    dialog.resolution.setCurrentIndex(dialog.resolution.findData(72))
    assert dialog.values() == (842, 595, "white", 72.0)

    dialog.resolution.setCurrentIndex(dialog.resolution.findData(150))
    dialog.template.setCurrentIndex(dialog.template.findData("a3"))
    assert dialog.values()[:2] == (2480, 1754)
    assert dialog.values()[3] == 150.0

    dialog.template.setCurrentIndex(dialog.template.findData("a4"))
    dialog.orientation.setCurrentIndex(dialog.orientation.findData("portrait"))
    assert dialog.values()[:2] == (1240, 1754)
    dialog.width.setValue(100)
    assert dialog.template.currentData() == "custom"
    assert dialog.values() == (100, 1754, "white", 150.0)
    assert not dialog.orientation.isEnabled()
    assert dialog.resolution.isEnabled()
    assert dialog.summary.text() == "100 × 1754 px"

    dialog.template.setCurrentIndex(dialog.template.findData("hd"))
    assert dialog.values() == (1920, 1080, "white", 72.0)
    assert not dialog.orientation.isEnabled()
    assert not dialog.resolution.isEnabled()
    assert dialog.summary.text() == "1920 × 1080 px"

    dialog.template.setCurrentIndex(dialog.template.findData("custom"))
    assert dialog.values() == (1280, 800, "white", 72.0)
    assert dialog.resolution.isEnabled()
