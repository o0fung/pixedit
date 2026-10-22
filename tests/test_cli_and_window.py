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
        assert tool_id == "example.marker" or button.text()
        tip = button.toolTip()
        tool = window.tool_by_id[tool_id]
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
