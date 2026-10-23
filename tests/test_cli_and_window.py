"""Console script and a headless window smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pixedit.cli import main


def test_version_and_help():
    with pytest.raises(SystemExit) as version:
        main(["--version"])
    assert version.value.code == 0
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0


def test_check_constructs_window():
    assert main(["--check"]) == 0


def test_window_menus_plugins_and_open(tmp_path: Path):
    from pixedit.ui.main_window import MainWindow
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    titles = [action.text() for action in window.menuBar().actions()]
    assert "File" in titles and "Layer" in titles and "Filter" in titles
    labels = []
    for action in window.menuBar().actions():
        menu = action.menu()
        if menu is None:
            continue
        labels.extend(child.text() for child in menu.actions())
    assert any("Vignette" in label for label in labels)
    assert any("Trim" in label for label in labels)
    assert any("Clear selection contents" in label for label in labels)
    assert "example.marker" in window.tool_by_id
    assert "redact" in window.tool_by_id
    for tool_id, button in window._buttons.items():
        assert not button.icon().isNull(), tool_id
        tool = window.tool_by_id[tool_id]
        assert button.text() == tool.button_caption()
        if tool.shortcut:
            assert button.text() == f"{tool.label} ({tool.shortcut})"
        else:
            assert button.text() == tool.label
        tip = button.toolTip()
        assert tool.label in tip
        if tool.shortcut:
            assert tool.shortcut in tip
        if tool.hint:
            assert tool.hint in tip
    path = tmp_path / "photo.png"
    Image.new("RGBA", (12, 9), (20, 30, 40, 255)).save(path)
    window.open_path(str(path))
    app.processEvents()
    assert window.session.document.width == 12
    assert window.session.document.layers[0].image.getpixel((0, 0))[:3] == (20, 30, 40)
    window.session.undo()
    window.session.redo()
    window.close()
    app.processEvents()


def test_canvas_mouse_move_reports_cursor():
    """A hover must report canvas coordinates. Passing ``pos.y`` without calling it aborted the process."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    app.processEvents()
    local = QPointF(120, 90)
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        local,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window.canvas, move)
    app.processEvents()
    expected_x, expected_y = window.canvas.widget_to_canvas(local.x(), local.y())
    assert window._cursor == pytest.approx((expected_x, expected_y))
    assert f"{expected_x:.0f}, {expected_y:.0f}" in window._status_label.text()
    window.close()
    app.processEvents()


def test_delete_clears_selection_pixels():
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication

    from pixedit.tools.base import PointerEvent
    from pixedit.tools.builtins import SelectTool
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    app.processEvents()
    window.session.new_canvas(24, 24, "white")
    window.session.options["select_mode"] = "rect"
    tool = SelectTool()
    tool.pointer_down(window.session, PointerEvent(4, 4))
    tool.pointer_up(window.session, PointerEvent(12, 12))
    window.canvas.setFocus()
    app.processEvents()
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(window.canvas, event)
    app.processEvents()
    image = window.session.active_layer().image
    assert image.getpixel((6, 6))[3] == 0
    assert image.getpixel((1, 1))[3] == 255
    assert image.size == (24, 24)
    window.session.dirty = False
    window.close()
    app.processEvents()


def _menu(window, title: str):
    for action in window.menuBar().actions():
        if action.text() == title and action.menu() is not None:
            return action.menu()
    raise AssertionError(title)


def test_file_toolbar_and_tool_groups():
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(900, 700)
    window.show()
    app.processEvents()
    actions = [action for action in window.file_toolbar.actions() if not action.isSeparator()]
    assert [action.iconText() for action in actions] == ["Save", "Open", "Place", "Export"]
    assert [action.text() for action in actions] == [
        "Save project",
        "Open…",
        "Place…",
        "Export flattened…",
    ]
    file_menu = _menu(window, "File")
    by_text = {action.text(): action for action in file_menu.actions()}
    for action in actions:
        assert by_text[action.text()] is action
        assert not action.icon().isNull()
        assert "Ctrl" in action.toolTip()
    assert "Ctrl+S" in window._save_action.toolTip()
    assert "Ctrl+O" in window._open_action.toolTip()
    assert "Ctrl+Shift+O" in window._place_action.toolTip()
    assert "Ctrl+E" in window._export_action.toolTip()
    assert window.file_toolbar.isVisible()
    buttons = [button for button in window.file_toolbar.findChildren(QToolButton) if button.text()]
    assert {button.text() for button in buttons} == {"Save", "Open", "Place", "Export"}

    layer_labels = [action.text() for action in _menu(window, "Layer").actions()]
    select_labels = [action.text() for action in _menu(window, "Select").actions()]
    edit_labels = [action.text() for action in _menu(window, "Edit").actions()]
    for label in ("Cut selection to new layer", "Copy selection to new layer"):
        assert label in edit_labels
        assert label in select_labels
        assert label in layer_labels
    assert "Deselect" in edit_labels
    assert "Deselect" in select_labels
    assert "Place image…" in layer_labels
    assert "Place…" in [action.text() for action in file_menu.actions()]

    order = []
    for index in range(window.tool_layout.count()):
        widget = window.tool_layout.itemAt(index).widget()
        if widget is None:
            continue
        if widget.objectName() == "toolSeparator":
            order.append("|")
        elif isinstance(widget, QToolButton):
            order.append(widget.text())
    assert order == [
        "Move (V)",
        "Hand (H)",
        "Zoom (Z)",
        "Select (M)",
        "Eyedrop (I)",
        "|",
        "Brush (B)",
        "Eraser (E)",
        "Fill (G)",
        "Shape (U)",
        "Text (T)",
        "|",
        "Crop (C)",
        "Redact (R)",
        "|",
        "Marker",
    ]
    window.close()
    app.processEvents()


def test_opacity_slider_and_brush_cursor():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.canvas import brush_circle_cursor
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    cursor = brush_circle_cursor(30)
    pixmap = cursor.pixmap()
    assert not pixmap.isNull()
    ratio = pixmap.devicePixelRatio() or 1.0
    assert pixmap.width() / ratio >= 30
    hotspot = cursor.hotSpot()
    assert abs(hotspot.x() - (pixmap.width() / ratio) / 2) <= 2
    assert abs(hotspot.y() - (pixmap.height() / ratio) / 2) <= 2

    window = MainWindow()
    window.resize(900, 700)
    window.show()
    app.processEvents()
    window.opacity_slider.setValue(40)
    app.processEvents()
    assert window.session.foreground[3] == round(40 * 255 / 100)
    assert "40%" in window.swatch.text()
    assert "40%" in window.opacity_label.text()

    window.set_tool(window.tool_by_id["eraser"])
    window.session.options["brush_size"] = 40
    window.canvas.zoom = 2.0
    window.canvas.refresh_tool_cursor()
    brush_cursor = window.canvas.cursor()
    brush_pixmap = brush_cursor.pixmap()
    brush_ratio = brush_pixmap.devicePixelRatio() or 1.0
    assert brush_pixmap.width() / brush_ratio >= 70
    window.set_tool(window.tool_by_id["move"])
    assert window.canvas.cursor().pixmap().isNull()

    window.set_tool(window.tool_by_id["brush"])
    window.session.options["brush_size"] = 10
    window.canvas.zoom = 2.0
    window.canvas.refresh_tool_cursor()
    small = window.canvas.cursor().pixmap().width()
    window._bump_brush(30)
    large = window.canvas.cursor().pixmap().width()
    assert large > small + 20
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_escape_deselects_when_idle_and_cancels_a_shape_first():
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication

    from pixedit.tools.base import PointerEvent
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    app.processEvents()
    window.session.new_canvas(40, 40, "transparent")
    from PIL import Image

    mask = Image.new("L", (40, 40), 0)
    mask.paste(255, (4, 4, 16, 16))
    window.session.set_selection(mask)
    window.set_tool(window.tool_by_id["shape"])
    window.session.foreground = (255, 0, 0, 255)
    window.session.options["shape_kind"] = "rect"
    window.session.options["shape_filled"] = True
    shape = window.current
    shape.pointer_down(window.session, PointerEvent(8, 8))
    shape.pointer_move(window.session, PointerEvent(28, 24))
    # The selection clips the preview, so sample a pixel that actually painted.
    layer = window.session.active_layer()
    box = layer.image.getchannel("A").getbbox()
    assert box is not None
    sample_x = layer.x + box[0]
    sample_y = layer.y + box[1]
    assert layer.image.getpixel((box[0], box[1]))[0] > 200
    window.canvas.setFocus()
    app.processEvents()
    escape = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(window.canvas, escape)
    app.processEvents()
    assert window.session.document.selection is not None
    assert window.session.active_layer().image.getpixel((sample_x, sample_y))[3] == 0
    QApplication.sendEvent(window.canvas, escape)
    app.processEvents()
    assert window.session.document.selection is None
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_open_image_brush_and_fill_with_and_without_a_layer_selection(tmp_path: Path):
    from PyQt6.QtWidgets import QApplication

    from pixedit.tools.base import PointerEvent
    from pixedit.tools.builtins import BrushTool, FillTool
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(900, 700)
    window.show()
    app.processEvents()
    path = tmp_path / "photo.png"
    Image.new("RGBA", (32, 24), (255, 255, 255, 255)).save(path)
    window.open_path(str(path))
    app.processEvents()
    assert window.session.layer_index == 0
    assert window.layers.list.currentRow() == 0

    window.session.foreground = (255, 0, 0, 255)
    window.session.options["brush_size"] = 6
    window.session.options["brush_hardness"] = 100
    window.session.options["brush_opacity"] = 100
    brush = BrushTool()
    brush.pointer_down(window.session, PointerEvent(8, 8))
    brush.pointer_up(window.session, PointerEvent(8, 8))
    assert window.session.active_layer().image.getpixel((8, 8))[0] > 200

    window.layers.list.setCurrentRow(-1)
    app.processEvents()
    assert window.layers.list.currentRow() == -1
    assert window.session.layer_index == -1
    assert window.session.active_layer() is None

    window.session.foreground = (0, 0, 255, 255)
    window.session.options["tolerance"] = 0
    FillTool().pointer_down(window.session, PointerEvent(16, 12))
    app.processEvents()
    photo = window.session.document.layers[0]
    assert photo.image.getpixel((16, 12))[:3] == (0, 0, 255)
    assert window.session.layer_index == 0
    assert window.layers.list.currentRow() == 0
    assert len(window.session.document.layers) == 1
    window.session.dirty = False
    window.close()
    app.processEvents()
