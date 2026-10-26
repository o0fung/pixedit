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


def test_status_version_and_about_guide():
    """The version stays on the status bar, and About includes the short guide."""
    from PyQt6.QtWidgets import QApplication

    from pixedit import __version__
    from pixedit.ui.dialogs import AboutDialog
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert window._version_label.text() == __version__
    dialog = AboutDialog(window)
    assert dialog.version.text() == __version__
    guide = dialog.body.toPlainText()
    assert "Save writes a" in guide
    assert ".pix" in guide
    window.close()
    app.processEvents()


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
    assert "mask" in window.tool_by_id
    for tool_id, button in window._buttons.items():
        assert not button.icon().isNull(), tool_id
        tool = window.tool_by_id[tool_id]
        # The button stores && so Qt draws a literal ampersand.
        visible = button.text().replace("&&", "&")
        assert visible == tool.button_caption()
        if tool.shortcut:
            assert visible == f"{tool.label} ({tool.shortcut})"
        else:
            assert visible == tool.label
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


def test_opened_file_keeps_its_name(tmp_path: Path):
    """An opened image names the window and the Save and Export suggestions.

    The project path stays empty, so Save still asks for a .pix file and does
    not overwrite the PNG. A new canvas goes back to Untitled.
    """
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    path = tmp_path / "photo.png"
    Image.new("RGBA", (12, 9), (20, 30, 40, 255)).save(path)
    window.open_path(str(path))
    app.processEvents()
    assert window.session.path is None
    assert window.session.source_name == "photo.png"
    assert window.windowTitle() == "PixEdit — photo.png"
    assert window._suggested_project_name() == "photo.pix"
    filename, selected = window._export_suggestion()
    assert filename == "photo.png"
    assert selected.startswith("PNG")

    window.session.source_name = "scan.pdf"
    window._update_title()
    assert window.windowTitle() == "PixEdit — scan.pdf"
    assert window._suggested_project_name() == "scan.pix"
    filename, selected = window._export_suggestion()
    assert filename == "scan.pdf"
    assert selected.startswith("PDF")

    window.session.path = str(tmp_path / "kept.pix")
    window._update_title()
    assert window.windowTitle() == "PixEdit — kept.pix"
    assert window._suggested_project_name() == "kept.pix"
    assert window._export_suggestion()[0] == "kept.png"

    window.session.new_canvas(8, 8, "white")
    app.processEvents()
    assert window.session.path is None
    assert window.session.source_name is None
    assert window.windowTitle() == "PixEdit — Untitled"
    assert window._suggested_project_name() == "untitled.pix"
    assert window._export_suggestion()[0] == "export.png"

    window.close()
    app.processEvents()


def test_tool_description_sits_under_the_options():
    """The description is the last row, under the settings and buttons."""
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    options = window.options
    layout = options.layout()
    assert layout.indexOf(options.heading) < layout.indexOf(options.form_host)
    assert layout.indexOf(options.form_host) < layout.indexOf(options.actions_host)
    hint_index = layout.indexOf(options.hint)
    assert hint_index == layout.count() - 1
    spacer = layout.itemAt(hint_index - 1)
    assert spacer is not None and spacer.spacerItem() is not None
    window.set_tool(window.tool_by_id["brush"])
    app.processEvents()
    assert options.hint.text()
    assert options.hint.isVisible()
    assert options.form_host.y() < options.hint.y()
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
    from pixedit.tools.select import SelectTool
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


def _shortcut_text(action, native: bool) -> str:
    from PyQt6.QtGui import QKeySequence

    form = QKeySequence.SequenceFormat.NativeText if native else QKeySequence.SequenceFormat.PortableText
    return action.shortcut().toString(form)


def test_file_toolbar_and_tool_groups():
    from PyQt6.QtGui import QKeySequence
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.output_folder import output_folder
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1400, 700)
    window.show()
    app.processEvents()
    kinds = ["|" if action.isSeparator() else action.iconText() for action in window.file_toolbar.actions()]
    assert kinds == [
        "New",
        "Open",
        "Close",
        "Save",
        "Save As",
        "Place",
        "Export",
        "|",
        "Undo",
        "Redo",
        "|",
        "Show folder",
        "About",
    ]
    actions = [action for action in window.file_toolbar.actions() if not action.isSeparator()]
    file_actions = [
        action
        for action in actions
        if action not in (window.undo_action, window.redo_action, window._about_action)
    ]
    assert [action.text() for action in file_actions] == [
        "New…",
        "Open…",
        "Close",
        "Save project",
        "Save project as…",
        "Place…",
        "Export flattened…",
        "Show output folder",
    ]
    file_menu = _menu(window, "File")
    by_text = {action.text(): action for action in file_menu.actions()}
    folder = str(output_folder())
    for action in file_actions:
        assert by_text[action.text()] is action
        assert not action.icon().isNull()
        if action is window._show_folder_action:
            assert folder in action.toolTip()
            assert action.shortcut().isEmpty()
        else:
            # The tooltip shows the platform shortcut, ⌘ on macOS and Ctrl elsewhere.
            assert _shortcut_text(action, native=True) in action.toolTip()
    visible = [action.text() for action in file_menu.actions() if action.text()]
    assert visible[-4:] == ["Show output folder", "Choose output folder…", "Close", "Quit"]
    help_menu = _menu(window, "Help")
    help_by_text = {action.text(): action for action in help_menu.actions()}
    assert help_by_text["About PixEdit"] is window._about_action
    assert window._about_action.iconText() == "About"
    assert not window._about_action.icon().isNull()
    assert window._about_action.toolTip() == "About PixEdit"
    assert window._choose_folder_action not in window.file_toolbar.actions()
    assert folder in window._choose_folder_action.toolTip()
    assert _shortcut_text(window._new_action, native=False) == "Ctrl+N"
    assert _shortcut_text(window._open_action, native=False) == "Ctrl+O"
    # Offscreen Qt reports Close as Ctrl+F4. The Mac app uses ⌘W.
    # Either string is StandardKey.Close.
    close_keys = QKeySequence(QKeySequence.StandardKey.Close).toString(QKeySequence.SequenceFormat.PortableText)
    assert _shortcut_text(window._close_action, native=False) == close_keys
    assert _shortcut_text(window._save_action, native=False) == "Ctrl+S"
    assert _shortcut_text(window._save_as_action, native=False) == "Ctrl+Shift+S"
    assert _shortcut_text(window._place_action, native=False) == "Ctrl+Shift+O"
    assert _shortcut_text(window._export_action, native=False) == "Ctrl+E"
    assert window.file_toolbar.isVisible()
    buttons = []
    for action in window.file_toolbar.actions():
        if action.isSeparator():
            continue
        buttons.extend(
            widget for widget in action.associatedObjects() if isinstance(widget, QToolButton) and widget.text()
        )
    assert {button.text() for button in buttons} == {
        "New",
        "Open",
        "Close",
        "Save",
        "Save As",
        "Place",
        "Export",
        "Undo",
        "Redo",
        "Show folder",
        "About",
    }
    assert not window.undo_action.icon().isNull()
    assert not window.redo_action.icon().isNull()
    assert window.undo_action.iconText() == "Undo"
    assert window.redo_action.iconText() == "Redo"
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()
    assert _shortcut_text(window.undo_action, native=True) in window.undo_action.toolTip()
    assert _shortcut_text(window.redo_action, native=True) in window.redo_action.toolTip()
    edit_by_text = {action.text(): action for action in _menu(window, "Edit").actions()}
    assert edit_by_text["Undo"] is window.undo_action
    assert edit_by_text["Redo"] is window.redo_action
    # The menu names the step. The button stays the short word.
    window.session.set_opacity(window.session.active_layer(), 0.4)
    app.processEvents()
    assert window.undo_action.text() == "Undo Layer opacity"
    assert window.undo_action.iconText() == "Undo"
    assert window.undo_action.isEnabled()
    assert "Layer opacity" in window.undo_action.toolTip()
    buttons = []
    for action in window.file_toolbar.actions():
        if action.isSeparator():
            continue
        buttons.extend(
            widget for widget in action.associatedObjects() if isinstance(widget, QToolButton) and widget.text()
        )
    assert "Undo" in {button.text() for button in buttons}
    assert "Undo Layer opacity" not in {button.text() for button in buttons}
    window.session.undo()
    app.processEvents()
    assert not window.undo_action.isEnabled()
    assert window.redo_action.text() == "Redo Layer opacity"
    assert window.redo_action.iconText() == "Redo"
    assert window.redo_action.isEnabled()

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
    assert "Apply mask" in layer_labels
    assert "Place…" in [action.text() for action in file_menu.actions()]

    order = []
    for index in range(window.tool_layout.count()):
        widget = window.tool_layout.itemAt(index).widget()
        if widget is None:
            continue
        if widget.objectName() == "toolSeparator":
            order.append("|")
        elif isinstance(widget, QToolButton):
            order.append(widget.text().replace("&&", "&"))
    assert order == [
        "Transform (V)",
        "Hand (H)",
        "Zoom (Z)",
        "Select (S)",
        "|",
        "Brush (B)",
        "Eraser (E)",
        "Fill (G)",
        "Shape (U)",
        "Text (T)",
        "|",
        "Image & Eyedrop (I)",
        "Sizes & Crop (C)",
        "Mask (R)",
        "|",
        "Marker",
    ]
    window.session.dirty = False
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
    window.palette.opacity_slider.setValue(40)
    app.processEvents()
    assert window.session.foreground[3] == round(40 * 255 / 100)
    assert "40%" in window.palette.opacity_label.text()
    assert window.palette.hex_edit.text().startswith("#")

    window.set_tool(window.tool_by_id["eraser"])
    window.session.options["brush_size"] = 40
    window.options.sync("brush_size", 40)
    assert window.options.brush_preview is not None
    eraser_preview = window.options.brush_preview.pixmap().toImage()
    assert eraser_preview.pixelColor(2, 2).alpha() > eraser_preview.pixelColor(88, 48).alpha()
    window.canvas.zoom = 2.0
    window.canvas.refresh_tool_cursor()
    brush_cursor = window.canvas.cursor()
    brush_pixmap = brush_cursor.pixmap()
    brush_ratio = brush_pixmap.devicePixelRatio() or 1.0
    assert brush_pixmap.width() / brush_ratio >= 70
    window.set_tool(window.tool_by_id["transform"])
    assert window.canvas.cursor().pixmap().isNull()
    assert window.options.brush_preview is None

    window.set_tool(window.tool_by_id["brush"])
    window.session.foreground = (255, 0, 0, 255)
    window.options._widgets["brush_size"].setValue(48)
    window.options._widgets["brush_hardness"].setValue(100)
    app.processEvents()
    assert window.options.brush_preview is not None
    hard_red = window.options.brush_preview.pixmap().toImage().pixelColor(108, 48).red()
    window.options._widgets["brush_hardness"].setValue(0)
    app.processEvents()
    soft_red = window.options.brush_preview.pixmap().toImage().pixelColor(108, 48).red()
    assert hard_red > soft_red + 40
    window.session.foreground = (20, 24, 28, 255)
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


def _cursor_ink(cursor) -> bytes:
    from PyQt6.QtGui import QImage

    image = cursor.pixmap().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    bits = image.constBits()
    bits.setsize(image.sizeInBytes())
    return bytes(bits)


def _hotspot_is_ink(cursor) -> bool:
    pixmap = cursor.pixmap()
    image = pixmap.toImage()
    ratio = pixmap.devicePixelRatio() or 1.0
    hot = cursor.hotSpot()
    x = min(image.width() - 1, max(0, int(round(hot.x() * ratio))))
    y = min(image.height() - 1, max(0, int(round(hot.y() * ratio))))
    return image.pixelColor(x, y).alpha() > 200


def test_tool_cursors_match_the_tool():
    """Each tool's pointer is the glyph for that action, and the click lands on ink."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from pixedit.tools.base import Tool
    from pixedit.ui.cursors import GLYPH_CURSORS, tool_pixmap_cursor
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    glyphs = {}
    for name in GLYPH_CURSORS:
        for scale in (1.0, 2.0):
            cursor = tool_pixmap_cursor(name, device_pixel_ratio=scale)
            assert cursor is not None, name
            assert _hotspot_is_ink(cursor), name
        glyphs[name] = _cursor_ink(tool_pixmap_cursor(name, device_pixel_ratio=1.0))
    assert tool_pixmap_cursor("sparkle") is None
    assert glyphs["shape-rect"] != glyphs["shape-ellipse"]
    assert glyphs["shape-rect"] != glyphs["select-rect"]
    assert glyphs["zoom-in"] != glyphs["zoom-out"]

    window = MainWindow()
    window.resize(900, 700)
    window.show()
    app.processEvents()
    scale = float(window.canvas.devicePixelRatioF() or 1.0)

    def glyph(name: str) -> bytes:
        return _cursor_ink(tool_pixmap_cursor(name, device_pixel_ratio=scale))

    def ink(tool_id: str) -> bytes:
        window.set_tool(window.tool_by_id[tool_id])
        app.processEvents()
        return _cursor_ink(window.canvas.cursor())

    assert ink("fill") == glyph("bucket")
    assert ink("color") == glyph("eyedrop")
    assert ink("crop") == glyph("crop")
    assert ink("zoom") == glyph("zoom-in")

    window.canvas.setFocus()
    QTest.keyPress(window.canvas, Qt.Key.Key_Alt)
    app.processEvents()
    assert _cursor_ink(window.canvas.cursor()) == glyph("zoom-out")
    QTest.keyRelease(window.canvas, Qt.Key.Key_Alt)
    app.processEvents()
    assert _cursor_ink(window.canvas.cursor()) == glyph("zoom-in")

    window.set_tool(window.tool_by_id["shape"])
    shape = window.options._widgets["shape_kind"]

    def pick(widget, value: str) -> None:
        QTest.mouseClick(widget._by_value[value], Qt.MouseButton.LeftButton)
        app.processEvents()

    pick(shape, "ellipse")
    assert _cursor_ink(window.canvas.cursor()) == glyph("shape-ellipse")
    pick(shape, "line")
    assert _cursor_ink(window.canvas.cursor()) == glyph("shape-line")

    window.set_tool(window.tool_by_id["select"])
    mode = window.options._widgets["select_mode"]
    pick(mode, "wand")
    assert _cursor_ink(window.canvas.cursor()) == glyph("wand")
    pick(mode, "freehand")
    assert _cursor_ink(window.canvas.cursor()) == glyph("lasso")

    window.set_tool(window.tool_by_id["mask"])
    mask = window.options._widgets["mask_shape"]
    pick(mask, "ellipse")
    assert _cursor_ink(window.canvas.cursor()) == glyph("select-ellipse")

    window.set_tool(window.tool_by_id["brush"])
    assert not window.canvas.cursor().pixmap().isNull()
    window.set_tool(window.tool_by_id["eraser"])
    assert not window.canvas.cursor().pixmap().isNull()
    window.set_tool(window.tool_by_id["text"])
    assert window.canvas.cursor().pixmap().isNull()
    assert window.canvas.cursor().shape() == Qt.CursorShape.IBeamCursor
    window.set_tool(window.tool_by_id["transform"])
    assert window.canvas.cursor().pixmap().isNull()
    assert window.canvas.cursor().shape() == Qt.CursorShape.SizeAllCursor
    rotate = tool_pixmap_cursor("rotate", device_pixel_ratio=1.0)
    assert _hotspot_is_ink(rotate)
    rotate_pixmap = rotate.pixmap()
    rotate_ratio = rotate_pixmap.devicePixelRatio() or 1.0
    rotate_center = (rotate_pixmap.width() / rotate_ratio) / 2.0
    assert abs(rotate.hotSpot().x() - rotate_center) <= 2
    assert abs(rotate.hotSpot().y() - rotate_center) <= 2
    window.set_tool(window.tool_by_id["hand"])
    assert window.canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor

    window.set_tool(window.tool_by_id["fill"])
    QTest.keyPress(window.canvas, Qt.Key.Key_Space)
    app.processEvents()
    assert window.canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor
    QTest.keyRelease(window.canvas, Qt.Key.Key_Space)
    app.processEvents()
    assert _cursor_ink(window.canvas.cursor()) == glyph("bucket")

    class _OddTool(Tool):
        id = "odd"
        cursor = "sparkle"

    window.current = _OddTool()
    window.canvas.refresh_tool_cursor()
    assert window.canvas.cursor().shape() == Qt.CursorShape.ArrowCursor

    window.set_tool(window.tool_by_id["transform"])
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


def test_parse_hex():
    from pixedit.ui.palette import parse_hex

    assert parse_hex("#0F0") == (0, 255, 0, None)
    assert parse_hex("#112233") == (0x11, 0x22, 0x33, None)
    assert parse_hex("11223380") == (0x11, 0x22, 0x33, 0x80)
    assert parse_hex("nope") is None


def _mouse_click(widget, x: float, y: float) -> None:
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QApplication

    pos = QPointF(x, y)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, press)
    QApplication.sendEvent(widget, release)


def test_color_code_box_returns_keyboard_to_the_canvas():
    """The code box keeps letters only while it is being edited.

    Enter applies the code and returns to the picture, so the next letter is
    a tool shortcut. Escape throws the typed text away. A click on the color
    square does the same. Changing tools returns to the picture when the
    keyboard was in the code box or in the options panel, and leaves the
    layer list alone.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    window.activateWindow()
    app.processEvents()

    code = window.palette.hex_edit
    code.setFocus(Qt.FocusReason.MouseFocusReason)
    code.setText("#00FF00")
    QTest.keyClick(code, Qt.Key.Key_Return)
    app.processEvents()
    assert window.session.foreground[:3] == (0, 255, 0)
    assert window.focusWidget() is window.canvas
    assert code.text().upper() == "#00FF00"

    QTest.keyClick(window.canvas, Qt.Key.Key_B)
    app.processEvents()
    assert window.current.id == "brush"
    assert code.text().upper() == "#00FF00"

    code.setFocus(Qt.FocusReason.MouseFocusReason)
    code.setText("nope")
    QTest.keyClick(code, Qt.Key.Key_Escape)
    app.processEvents()
    assert window.session.foreground[:3] == (0, 255, 0)
    assert code.text().upper() == "#00FF00"
    assert window.focusWidget() is window.canvas

    code.setFocus(Qt.FocusReason.MouseFocusReason)
    code.setText("#0000FF")
    pad = window.palette.sv_pad
    pad.resize(100, 120)
    app.processEvents()
    _mouse_click(pad, pad.width() - 1, 1)
    app.processEvents()
    assert window.focusWidget() is window.canvas
    assert window.session.foreground[:3] != (0, 0, 255)
    assert code.text().upper() != "#0000FF"

    code.setFocus(Qt.FocusReason.MouseFocusReason)
    app.processEvents()
    assert window.focusWidget() is code
    code.setFocus(Qt.FocusReason.TabFocusReason)
    app.processEvents()
    assert window.focusWidget() is code

    window.canvas.setFocus()
    code.setFocus(Qt.FocusReason.OtherFocusReason)
    app.processEvents()
    assert window.focusWidget() is window.canvas

    code.setFocus(Qt.FocusReason.MouseFocusReason)
    app.processEvents()
    window.set_tool(window.tool_by_id["eraser"])
    app.processEvents()
    assert window.current.id == "eraser"
    assert window.focusWidget() is window.canvas

    window.set_tool(window.tool_by_id["brush"])
    app.processEvents()
    slider = window.options._widgets["brush_size"]
    slider.setFocus(Qt.FocusReason.MouseFocusReason)
    app.processEvents()
    assert window.focusWidget() is slider
    window.set_tool(window.tool_by_id["eraser"])
    app.processEvents()
    assert window.focusWidget() is window.canvas
    assert window.focusWidget() is not code

    window.layers.list.setFocus(Qt.FocusReason.MouseFocusReason)
    app.processEvents()
    assert window.focusWidget() is window.layers.list
    window.set_tool(window.tool_by_id["transform"])
    app.processEvents()
    assert window.focusWidget() is window.layers.list

    window.session.dirty = False
    window.close()
    app.processEvents()


def test_space_pans_from_the_layer_list_without_hiding_layers():
    """Space stays the hand tool. It does not flip a layer checkbox or the Visibility button."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QAbstractButton, QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    app.processEvents()
    layer = window.session.active_layer()
    assert layer.visible is True
    for button in window.layers.findChildren(QAbstractButton):
        assert button.focusPolicy() == Qt.FocusPolicy.NoFocus

    window.layers.list.setFocus(Qt.FocusReason.MouseFocusReason)
    app.processEvents()
    assert window.focusWidget() is window.layers.list
    QTest.keyPress(window.layers.list, Qt.Key.Key_Space)
    app.processEvents()
    assert layer.visible is True
    assert window.canvas._space is True
    assert window.canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor
    QTest.keyRelease(window.layers.list, Qt.Key.Key_Space)
    app.processEvents()
    assert window.canvas._space is False
    assert layer.visible is True

    visibility = next(
        button for button in window.layers.findChildren(QAbstractButton) if button.accessibleName() == "Visibility"
    )
    visibility.click()
    app.processEvents()
    assert layer.visible is False
    assert window.focusWidget() is not visibility
    QTest.keyPress(window.focusWidget(), Qt.Key.Key_Space)
    app.processEvents()
    assert layer.visible is False
    assert window.canvas._space is True
    QTest.keyRelease(window.focusWidget(), Qt.Key.Key_Space)
    app.processEvents()
    assert window.canvas._space is False

    code = window.palette.hex_edit
    code.setFocus(Qt.FocusReason.MouseFocusReason)
    app.processEvents()
    QTest.keyClick(code, Qt.Key.Key_Space)
    app.processEvents()
    assert window.canvas._space is False
    assert " " in code.text()

    window.session.dirty = False
    window.close()
    app.processEvents()


def test_palette_text_select_and_zoom_panels():
    from PyQt6.QtWidgets import QApplication

    from pixedit.model import TextInfo
    from pixedit.tools.base import PointerEvent
    from pixedit.tools.color import ColorTool
    from pixedit.ui.dialogs import TextDialog
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    app.processEvents()

    ColorTool().pointer_down(window.session, PointerEvent(4, 4))
    app.processEvents()
    assert window.session.foreground[:3] == (255, 255, 255)
    assert window.palette.hex_edit.text().upper() == "#FFFFFF"

    pad = window.palette.sv_pad
    pad.resize(100, 120)
    app.processEvents()
    window.palette.set_hsv(0.0, 0.0, 0.0, transient=False)
    _mouse_click(pad, pad.width() - 1, 1)
    red, green, blue, _alpha = window.session.foreground
    assert red > 240 and green < 30 and blue < 30

    window.palette.note_recent()
    red_color = tuple(window.session.foreground)
    window.palette.set_hsv(0.33, 1.0, 1.0, transient=False)
    window.palette.note_recent()
    window.palette.recent_buttons[1].click()
    app.processEvents()
    assert tuple(window.session.foreground) == red_color

    window.palette.hex_edit.setText("#0000FF")
    window.palette.hex_edit.editingFinished.emit()
    assert window.session.foreground[:3] == (0, 0, 255)

    dialog = TextDialog(TextInfo("Hi", "/font.ttf", "Demo", 30, (1, 2, 3, 4), 12, "justify", 18))
    dialog.text.setPlainText("Yo")
    kept = dialog.result_info()
    assert kept.text == "Yo"
    assert kept.font_path == "/font.ttf"
    assert kept.font_family == "Demo"
    assert kept.size == 30
    assert kept.color == (1, 2, 3, 4)
    assert kept.rotation == 12
    assert kept.align == "justify"
    assert kept.line_spacing == 18

    window.session.add_text(TextInfo("Hi", None, "Default", 28, (255, 0, 0, 255), 200), 8, 8)
    window.set_tool(window.tool_by_id["text"])
    app.processEvents()
    assert window.options._labels["text_font"].text() == "Font"
    assert window.options._labels["text_style"].text() == "Style"
    assert window.options._labels["text_align"].text() == "Align"
    assert window.options._widgets["text_align"].labels() == ["Left", "Center", "Right", "Justify"]
    assert window.options._labels["text_size"].text() == "Size"
    assert window.options._labels["text_line_spacing"].text().startswith("Spacing")
    assert window.options._labels["text_rotation"].text().startswith("Rotation")
    assert "°" in window.options._labels["text_rotation"].text()
    assert window.session.options["text_rotation"] == 200
    assert window.session.active_layer().text.rotation == 200
    assert window.session.foreground[:3] == (255, 0, 0)
    steps = len(window.session.history.undo_stack)
    window.session.apply_text_style(transient=False)
    assert len(window.session.history.undo_stack) == steps
    layer = window.session.active_layer()
    center = (layer.x + layer.image.width / 2, layer.y + layer.image.height / 2)
    window.options._widgets["text_size"].setValue(72)
    app.processEvents()
    layer = window.session.active_layer()
    assert layer.text is not None and layer.text.size == 72
    assert abs((layer.x + layer.image.width / 2) - center[0]) <= 1
    assert abs((layer.y + layer.image.height / 2) - center[1]) <= 1
    window.session.undo()
    assert window.session.active_layer().text.size == 28

    window.palette.set_hsv(0.33, 1.0, 1.0, transient=False)
    app.processEvents()
    assert window.session.active_layer().text.color[1] > 200
    window.session.undo()
    assert window.session.active_layer().text.color[0] == 255

    window.set_tool(window.tool_by_id["brush"])
    kept_color = window.session.active_layer().text.color
    window.palette.set_hsv(0.0, 1.0, 1.0, transient=False)
    assert window.session.active_layer().text.color == kept_color
    assert window.session.foreground[0] > 240

    window.set_tool(window.tool_by_id["select"])
    app.processEvents()
    labels = [_button_label(button) for button in window.options.action_buttons]
    assert labels == [
        "Select all",
        "Deselect",
        "Invert",
        "Cut to layer",
        "Copy to layer",
        "Clear pixels",
        "Crop to selection",
    ]
    by_label = {_button_label(button): button for button in window.options.action_buttons}
    for button in window.options.action_buttons:
        assert not button.icon().isNull(), _button_label(button)
    assert by_label["Cut to layer"].toolTip() == "Cut to layer (Ctrl+X)"
    assert by_label["Crop to selection"].toolTip() == "Crop to selection"
    assert by_label["Select all"].isEnabled()
    assert not by_label["Deselect"].isEnabled()
    assert not by_label["Crop to selection"].isEnabled()
    window.session.select_all()
    app.processEvents()
    assert by_label["Deselect"].isEnabled()
    by_label["Deselect"].click()
    app.processEvents()
    assert window.session.document.selection is None

    from PIL import Image

    def _select_rect() -> None:
        window.session.new_canvas(20, 20, "white")
        mask = Image.new("L", (20, 20), 0)
        mask.paste(255, (2, 3, 10, 12))
        window.session.set_selection(mask)

    _select_rect()
    window._crop_selection()
    menu_size = (window.session.document.width, window.session.document.height)
    _select_rect()
    window.set_tool(window.tool_by_id["select"])
    by_label = {_button_label(button): button for button in window.options.action_buttons}
    assert by_label["Crop to selection"].isEnabled()
    by_label["Crop to selection"].click()
    app.processEvents()
    assert (window.session.document.width, window.session.document.height) == menu_size == (8, 9)

    window.set_tool(window.tool_by_id["zoom"])
    app.processEvents()
    zoom_labels = [_button_label(button) for button in window.options.action_buttons]
    assert zoom_labels == ["Zoom in", "Zoom out", "Fit", "100%"]
    zoom_buttons = {_button_label(button): button for button in window.options.action_buttons}
    for button in window.options.action_buttons:
        assert not button.icon().isNull(), _button_label(button)
    window.canvas.zoom = 2.0
    zoom_buttons["100%"].click()
    assert window.canvas.zoom == 1.0
    zoom_buttons["Zoom in"].click()
    assert window.canvas.zoom > 1.0
    zoom_buttons["Fit"].click()
    assert window.canvas.zoom > 0

    window.session.dirty = False
    window.close()
    app.processEvents()


def test_transform_color_and_mask_panels():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    app.processEvents()

    window.set_tool(window.tool_by_id["transform"])
    labels = [_button_label(button) for button in window.options.action_buttons]
    assert "Flip horizontal" in labels
    assert "Flip vertical" in labels
    assert "Rotate 90° clockwise" in labels
    assert "Rotate 90° counter-clockwise" in labels
    assert "Scale…" in labels
    assert "Rotate…" in labels
    buttons = {_button_label(button): button for button in window.options.action_buttons}
    for button in window.options.action_buttons:
        assert not button.icon().isNull(), _button_label(button)
    buttons["Flip horizontal"].click()
    app.processEvents()
    assert window.session.history.undo_stack[-1].name == "Flip horizontal"

    window.set_tool(window.tool_by_id["color"])
    apply = next(button for button in window.options.action_buttons if button.text() == "Apply")
    reset = next(button for button in window.options.action_buttons if button.text() == "Reset")
    assert not apply.isEnabled()
    assert not reset.isEnabled()

    window.set_tool(window.tool_by_id["crop"])
    crop_labels = [button.text() for button in window.options.action_buttons]
    assert "Canvas size…" in crop_labels
    assert "Image size…" in crop_labels
    assert "Trim transparent edges" in crop_labels
    assert "Crop to selection" in crop_labels
    assert not window.options._widgets["crop_ratio_w"].isVisible()
    window.options._widgets["crop_ratio"].setCurrentText("Custom")
    app.processEvents()
    assert window.options._widgets["crop_ratio_w"].isVisible()
    assert window.options._widgets["crop_ratio_h"].isVisible()
    image_labels = [action.text() for action in _menu(window, "Image").actions()]
    assert "Canvas size…" in image_labels
    assert "Image size…" in image_labels
    assert "72 dpi" in window._status_label.text()

    from pixedit.ui.dialogs import CanvasSizeDialog, ImageSizeDialog, ScaleDialog

    scale = ScaleDialog(8, 4)
    scale.form.width_percent.setValue(50)
    assert scale.pixel_size() == (4, 2)
    scale.form.keep.setChecked(False)
    scale.form.width_px.setValue(8)
    assert scale.pixel_size() == (8, 2)

    canvas_size = CanvasSizeDialog(10, 8, (1, 2, 3, 255))
    picker = canvas_size.anchor_picker
    assert picker.width() == picker.height()
    assert canvas_size.values()[2] == "c"
    placed = {}
    for button in picker._group.buttons():
        row, column, _row_span, _col_span = picker.layout().getItemPosition(picker.layout().indexOf(button))
        placed[button.property("anchor")] = (row, column)
        assert button.width() == button.height()
    assert placed == {
        "nw": (0, 0),
        "n": (0, 1),
        "ne": (0, 2),
        "w": (1, 0),
        "c": (1, 1),
        "e": (1, 2),
        "sw": (2, 0),
        "s": (2, 1),
        "se": (2, 2),
    }
    placed_buttons = {button.property("anchor"): button for button in picker._group.buttons()}
    placed_buttons["nw"].click()
    assert canvas_size.values()[2] == "nw"
    placed_buttons["e"].click()
    assert canvas_size.values()[2] == "e"

    image_size = ImageSizeDialog(100, 50, 72)
    image_size.resolution.setValue(144)
    assert image_size.values()[:2] == (200, 100)
    image_size.resample.setChecked(False)
    image_size.resolution.setValue(300)
    assert image_size.values()[:3] == (100, 50, 300.0)
    assert image_size.values()[4] is False

    window.set_tool(window.tool_by_id["mask"])
    combo = window.options._widgets["mask_kind"]
    assert [combo.itemText(index) for index in range(combo.count())] == [
        "Pixelate",
        "Blur",
        "Solid",
        "Noise",
        "Desaturate",
        "Invert",
        "Dim",
        "Sharpen",
    ]
    assert window.options._widgets["mask_amount"].isVisible()
    assert window.options._labels["mask_amount"].text().startswith("Block")
    combo.setCurrentText("Solid")
    app.processEvents()
    assert not window.options._widgets["mask_amount"].isVisible()
    combo.setCurrentText("Blur")
    app.processEvents()
    assert window.options._widgets["mask_amount"].isVisible()
    assert window.options._labels["mask_amount"].text().startswith("Radius")

    window.session.dirty = False
    window.close()
    app.processEvents()


def test_close_quits_instead_of_opening_a_canvas():
    """Close ends the app. A clean picture closes with no dialog and stays as it was."""
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    window.session.new_canvas(17, 19, "white")
    document = window.session.document
    assert not window.session.dirty
    assert "Close PixEdit" in window._close_action.toolTip()

    window._close_action.trigger()
    app.processEvents()

    assert not window.isVisible()
    assert window.session.document is document
    assert document.width == 17 and document.height == 19


def test_close_cancel_keeps_unsaved_picture(monkeypatch):
    """Unsaved work asks first. Cancel leaves the window and the picture in place."""
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from pixedit.ui import main_window as main_window_module
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    window.session.new_canvas(21, 13, "white")
    window.session.dirty = True
    document = window.session.document

    def cancel(*_args, **_kwargs):
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(main_window_module.QMessageBox, "question", cancel)
    window._close_action.trigger()
    app.processEvents()

    assert window.isVisible()
    assert window.session.document is document
    assert window.session.dirty
    assert document.width == 21 and document.height == 13
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_transform_buttons_stay_inside_the_tool_column():
    """Transform icons share a row in the default column and stay inside it when widened."""
    from PyQt6.QtWidgets import QSplitter

    from pixedit.app import create_app
    from pixedit.ui.main_window import MainWindow

    app = create_app()
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    app.processEvents()
    splitter = window.findChild(QSplitter)
    left = splitter.widget(0)
    assert left.width() == 248
    window.set_tool(window.tool_by_id["transform"])
    app.processEvents()
    buttons = {_button_label(button): button for button in window.options.action_buttons}

    def row_top(button) -> int:
        return button.mapTo(left, button.rect().topLeft()).y()

    # Four quarter-turn icons fit the default column. Scale and Rotate follow.
    assert row_top(buttons["Flip horizontal"]) == row_top(buttons["Flip vertical"])
    assert row_top(buttons["Flip horizontal"]) == row_top(buttons["Rotate 90° clockwise"])
    assert row_top(buttons["Flip horizontal"]) == row_top(buttons["Rotate 90° counter-clockwise"])
    assert row_top(buttons["Scale…"]) == row_top(buttons["Rotate…"])
    assert row_top(buttons["Scale…"]) != row_top(buttons["Flip horizontal"])
    _assert_buttons_inside(window.options.action_buttons, left)
    splitter.setSizes([420, 500, 280])
    app.processEvents()
    assert left.width() > 248
    buttons = {_button_label(button): button for button in window.options.action_buttons}
    assert row_top(buttons["Flip horizontal"]) == row_top(buttons["Flip vertical"])
    assert row_top(buttons["Rotate 90° clockwise"]) == row_top(buttons["Rotate 90° counter-clockwise"])
    assert row_top(buttons["Scale…"]) == row_top(buttons["Rotate…"])
    assert row_top(buttons["Scale…"]) != row_top(buttons["Flip horizontal"])
    _assert_buttons_inside(window.options.action_buttons, left)
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_text_options_follow_the_tool_column():
    """Text controls stay inside the tool column and grow when the splitter widens."""
    from PyQt6.QtWidgets import QSplitter

    from pixedit.app import create_app
    from pixedit.ui.main_window import MainWindow

    app = create_app()
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    app.processEvents()
    splitter = window.findChild(QSplitter)
    left = splitter.widget(0)
    assert left.width() == 248
    window.set_tool(window.tool_by_id["text"])
    app.processEvents()
    keys = (
        "text_font",
        "text_style",
        "text_align",
        "text_size",
        "text_line_spacing",
        "text_rotation",
    )
    widgets = {key: window.options._widgets[key] for key in keys}
    font = widgets["text_font"]
    assert font.toolTip() == font.currentText()
    _assert_options_inside(widgets, left)
    start = {key: widget.width() for key, widget in widgets.items()}

    splitter.setSizes([420, 500, 280])
    app.processEvents()
    assert left.width() > 248
    _assert_options_inside(widgets, left)
    assert widgets["text_font"].width() > start["text_font"] + 40
    assert widgets["text_line_spacing"].width() > start["text_line_spacing"] + 40
    wide_font = widgets["text_font"].width()
    wide_spacing = widgets["text_line_spacing"].width()

    splitter.setSizes([228, 700, 280])
    app.processEvents()
    assert left.width() == 228
    _assert_options_inside(widgets, left)
    assert widgets["text_font"].width() < wide_font
    assert widgets["text_line_spacing"].width() < wide_spacing

    # The closed box is the column. The open list is the longest family.
    font.showPopup()
    app.processEvents()
    view = font.view()
    text_width = view.sizeHintForColumn(0)
    popup = view.window()
    assert popup is not window
    assert popup.width() >= text_width
    assert view.minimumWidth() >= text_width
    font.hidePopup()
    app.processEvents()

    window.session.dirty = False
    window.close()
    app.processEvents()


def test_choice_icons_share_a_row_beside_their_label():
    """Mode, shape, mask, and align pictures stay on one row next to a short caption."""
    from PyQt6.QtWidgets import QSplitter

    from pixedit.app import create_app
    from pixedit.ui.main_window import MainWindow

    app = create_app()
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    app.processEvents()
    splitter = window.findChild(QSplitter)
    left = splitter.widget(0)
    assert left.width() == 248

    def check(tool_id: str, key: str, count: int, minimum: int) -> None:
        window.set_tool(window.tool_by_id[tool_id])
        app.processEvents()
        choice = window.options._widgets[key]
        assert len(choice.labels()) == count
        _assert_choice_one_row(choice, minimum=minimum)
        # The row spans the options content, so it is wider than the slider beside a long label.
        field = {
            "text": "text_size",
            "select": "tolerance",
            "shape": "stroke_width",
            "mask": "mask_amount",
        }[tool_id]
        assert choice.width() > window.options._widgets[field].width()

    for tool_id, key, count in (
        ("select", "select_mode", 4),
        ("shape", "shape_kind", 3),
        ("mask", "mask_shape", 3),
        ("text", "text_align", 4),
    ):
        check(tool_id, key, count, 33)
    assert window.options._labels["text_line_spacing"].text().startswith("Spacing")

    splitter.setSizes([228, 700, 280])
    app.processEvents()
    assert left.width() == 228
    for tool_id, key, count in (
        ("select", "select_mode", 4),
        ("shape", "shape_kind", 3),
        ("mask", "mask_shape", 3),
        ("text", "text_align", 4),
    ):
        check(tool_id, key, count, 28)

    window.session.dirty = False
    window.close()
    app.processEvents()


def _assert_options_inside(widgets, left) -> None:
    from pixedit.ui.tool_options import IconChoice

    widths = []
    for key, widget in widgets.items():
        top = widget.mapTo(left, widget.rect().topLeft())
        bottom = widget.mapTo(left, widget.rect().bottomRight())
        assert top.x() >= 0, key
        assert bottom.x() < left.width(), (key, bottom.x(), left.width())
        assert widget.width() > 0, key
        # Icon rows span the caption and the pictures. The other fields share one column.
        if not isinstance(widget, IconChoice):
            widths.append(widget.width())
    assert widths
    assert max(widths) - min(widths) <= 1


def _assert_choice_one_row(choice, minimum: int) -> None:
    """Pictures share one row, next to the caption, between ``minimum`` and 44px."""
    from PyQt6.QtWidgets import QToolButton

    buttons = sorted(choice.findChildren(QToolButton), key=lambda button: button.x())
    assert buttons
    assert len({button.y() for button in buttons}) == 1
    assert all(button.width() >= minimum for button in buttons)
    assert all(button.width() <= 44 for button in buttons)
    assert all(button.width() == button.height() for button in buttons)
    label = choice.label
    gap = buttons[0].x() - (label.x() + label.width())
    assert 0 <= gap <= 8, (label.text(), gap)
    for button in buttons:
        assert button.geometry().right() < choice.width(), button.accessibleName()
        assert button.geometry().bottom() < choice.height(), button.accessibleName()


def _button_label(button) -> str:
    return button.accessibleName() or button.text()


def _assert_buttons_inside(buttons, left) -> None:
    boxes = []
    for button in buttons:
        label = _button_label(button)
        top = button.mapTo(left, button.rect().topLeft())
        bottom = button.mapTo(left, button.rect().bottomRight())
        assert top.x() >= 0, label
        assert bottom.x() < left.width(), label
        assert bottom.y() > top.y(), label
        assert button.width() >= button.sizeHint().width(), label
        boxes.append((top.x(), top.y(), bottom.x(), bottom.y(), label))
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            separated = box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1]
            assert separated, (box[4], other[4])


            assert separated, (box[4], other[4])


_PICTORIAL_ICONS = (
    "flip-h",
    "flip-v",
    "rotate-cw",
    "rotate-ccw",
    "scale",
    "rotate",
    "zoom-out",
    "zoom-fit",
    "zoom-100",
    "align-left",
    "align-center",
    "align-right",
    "align-justify",
    "geom-rect",
    "geom-ellipse",
    "geom-line",
    "geom-freehand",
    "geom-wand",
    "select-rect",
    "select-ellipse",
    "select-freehand",
    "select-all",
    "deselect",
    "select-invert",
    "cut-to-layer",
    "copy-to-layer",
    "clear-pixels",
    "crop-to-selection",
    "layer-merge-down",
    "layer-new",
    "layer-copy",
    "layer-delete",
    "layer-up",
    "layer-down",
    "layer-visibility",
)


def test_pictorial_icons_load_and_choices_keep_their_values():
    """Picture buttons keep the old option ids, and every new icon file loads."""
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.ui.icons import icon_named
    from pixedit.ui.main_window import MainWindow

    for name in _PICTORIAL_ICONS:
        assert not icon_named(name).isNull(), name

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    app.processEvents()

    def menu_icons(title: str, labels: list[str]) -> None:
        by_text = {action.text(): action for action in _menu(window, title).actions()}
        for label in labels:
            assert not by_text[label].icon().isNull(), label

    menu_icons(
        "Image",
        [
            "Crop to selection",
            "Rotate canvas 90° clockwise",
            "Rotate canvas 90° counter-clockwise",
            "Flip canvas horizontal",
            "Flip canvas vertical",
        ],
    )
    image_actions = {action.text(): action for action in _menu(window, "Image").actions()}
    assert image_actions["Invert"].icon().isNull()
    menu_icons("Edit", ["Cut selection to new layer", "Copy selection to new layer", "Deselect", "Clear selection contents"])
    menu_icons("Layer", ["Merge down", "Scale…", "Rotate…", "Flip horizontal", "Flip vertical"])
    menu_icons("View", ["Zoom in", "Zoom out", "Fit"])
    menu_icons(
        "Select",
        [
            "All",
            "Deselect",
            "Invert",
            "Cut selection to new layer",
            "Copy selection to new layer",
            "Rectangular",
            "Elliptical",
            "Freehand",
            "Magic wand",
        ],
    )

    window.set_tool(window.tool_by_id["text"])
    app.processEvents()
    align = window.options._widgets["text_align"]
    assert align.labels() == ["Left", "Center", "Right", "Justify"]
    assert align.value() == "left"
    for button in align.findChildren(QToolButton):
        assert button.geometry().right() < align.width(), button.accessibleName()
        assert button.geometry().bottom() < align.height(), button.accessibleName()
        assert not button.icon().isNull(), button.accessibleName()
    next(button for button in align.findChildren(QToolButton) if button.accessibleName() == "Justify").click()
    app.processEvents()
    assert window.session.options["text_align"] == "justify"
    assert align.value() == "justify"

    window.set_tool(window.tool_by_id["shape"])
    app.processEvents()
    assert window.options._widgets["shape_kind"].labels() == ["Rectangle", "Ellipse", "Line"]
    assert window.options._widgets["shape_kind"].value() == "rect"

    window.set_tool(window.tool_by_id["select"])
    app.processEvents()
    assert window.options._widgets["select_mode"].labels() == [
        "Rectangular",
        "Elliptical",
        "Freehand",
        "Magic wand",
    ]

    window.set_tool(window.tool_by_id["mask"])
    app.processEvents()
    assert window.options._widgets["mask_shape"].labels() == ["Rectangle", "Ellipse", "Freehand"]

    window.set_tool(window.tool_by_id["transform"])
    app.processEvents()
    # A pair of command squares fits. A third does not, so the quarter-turns wrap.
    window.options._action_row_budget = lambda: 100
    window.options._rebuild_actions(window.options._action_specs)
    app.processEvents()
    rows = {_button_label(button): button.parentWidget() for button in window.options.action_buttons}
    assert rows["Flip horizontal"] is rows["Flip vertical"]
    assert rows["Rotate 90° clockwise"] is rows["Rotate 90° counter-clockwise"]
    assert rows["Flip horizontal"] is not rows["Rotate 90° clockwise"]
    assert rows["Scale…"] is rows["Rotate…"]
    assert rows["Scale…"] is not rows["Flip horizontal"]
    window.session.dirty = False
    window.close()
    app.processEvents()


_PICTORIAL_ICONS = (
    "flip-h",
    "flip-v",
    "rotate-cw",
    "rotate-ccw",
    "scale",
    "rotate",
    "zoom-out",
    "zoom-fit",
    "zoom-100",
    "align-left",
    "align-center",
    "align-right",
    "align-justify",
    "geom-rect",
    "geom-ellipse",
    "geom-line",
    "geom-freehand",
    "geom-wand",
    "select-rect",
    "select-ellipse",
    "select-freehand",
    "select-all",
    "deselect",
    "select-invert",
    "cut-to-layer",
    "copy-to-layer",
    "clear-pixels",
    "crop-to-selection",
    "layer-merge-down",
    "layer-new",
    "layer-copy",
    "layer-delete",
    "layer-up",
    "layer-down",
    "layer-visibility",
)


def test_pictorial_icons_load_and_choices_apply():
    """Every picture file loads, menus reuse it, and a choice button stores its id."""
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.ui.icons import icon_named
    from pixedit.ui.main_window import MainWindow

    for name in _PICTORIAL_ICONS:
        assert not icon_named(name).isNull(), name

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1100, 800)
    window.show()
    app.processEvents()

    def menu_icons(title: str, labels: tuple[str, ...]) -> None:
        actions = {action.text(): action for action in _menu(window, title).actions()}
        for label in labels:
            assert not actions[label].icon().isNull(), label

    menu_icons(
        "Image",
        (
            "Crop to selection",
            "Rotate canvas 90° clockwise",
            "Rotate canvas 90° counter-clockwise",
            "Flip canvas horizontal",
            "Flip canvas vertical",
        ),
    )
    image_actions = {action.text(): action for action in _menu(window, "Image").actions()}
    assert image_actions["Invert"].icon().isNull()
    menu_icons("Edit", ("Cut selection to new layer", "Copy selection to new layer", "Deselect", "Clear selection contents"))
    menu_icons("Layer", ("Merge down", "Scale…", "Rotate…", "Flip horizontal", "Flip vertical"))
    menu_icons("View", ("Zoom in", "Zoom out", "Fit"))
    menu_icons(
        "Select",
        (
            "All",
            "Deselect",
            "Invert",
            "Cut selection to new layer",
            "Copy selection to new layer",
            "Rectangular",
            "Elliptical",
            "Freehand",
            "Magic wand",
        ),
    )

    window.set_tool(window.tool_by_id["text"])
    app.processEvents()
    align = window.options._widgets["text_align"]
    assert align.labels() == ["Left", "Center", "Right", "Justify"]
    for button in align.findChildren(QToolButton):
        assert not button.icon().isNull(), button.accessibleName()
        assert button.geometry().right() < align.width(), button.accessibleName()
        assert button.geometry().bottom() < align.height(), button.accessibleName()
    justify = next(button for button in align.findChildren(QToolButton) if button.accessibleName() == "Justify")
    justify.click()
    app.processEvents()
    assert window.session.options["text_align"] == "justify"
    assert align.value() == "justify"

    window.set_tool(window.tool_by_id["shape"])
    app.processEvents()
    assert window.options._widgets["shape_kind"].labels() == ["Rectangle", "Ellipse", "Line"]
    window.set_tool(window.tool_by_id["select"])
    app.processEvents()
    assert window.options._widgets["select_mode"].labels() == ["Rectangular", "Elliptical", "Freehand", "Magic wand"]
    window.set_tool(window.tool_by_id["mask"])
    app.processEvents()
    assert window.options._widgets["mask_shape"].labels() == ["Rectangle", "Ellipse", "Freehand"]

    window.session.dirty = False
    window.close()
    app.processEvents()


def test_checker_stays_inside_the_canvas():
    """12px checker cells used to run past the right and bottom of the image."""
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QApplication

    from pixedit.session import Session
    from pixedit.ui.canvas import CanvasView

    app = QApplication.instance() or QApplication([])
    # 100×80 at zoom 2 is 200×160 screen pixels, not a multiple of the 12px cell.
    session = Session()
    session.new_canvas(100, 80, "white")
    canvas = CanvasView(session, lambda: None, lambda *_args: None)
    canvas.resize(480, 360)
    canvas.zoom = 2.0
    canvas.offset_x = 40.4
    canvas.offset_y = 36.7
    canvas.show()
    app.processEvents()

    light = QColor("#2a3038")
    outside, inside = _checker_pixels(canvas, light)
    assert outside == []

    session.new_canvas(100, 80, "transparent")
    canvas.invalidate()
    app.processEvents()
    outside, inside = _checker_pixels(canvas, light)
    assert outside == []
    assert inside > 0
    canvas.close()
    app.processEvents()


def _checker_pixels(canvas, light) -> tuple[list[tuple[int, int]], int]:
    """Light checker pixels whose centers sit outside the canvas, and the count inside."""
    image = canvas.grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    ruler = canvas.ruler()
    doc = canvas.session.document
    left = ruler + canvas.offset_x
    top = ruler + canvas.offset_y
    right = left + doc.width * canvas.zoom
    bottom = top + doc.height * canvas.zoom
    outside: list[tuple[int, int]] = []
    inside = 0
    for iy in range(image.height()):
        for ix in range(image.width()):
            color = image.pixelColor(ix, iy)
            if color.red() != light.red() or color.green() != light.green() or color.blue() != light.blue():
                continue
            lx = (ix + 0.5) / ratio
            ly = (iy + 0.5) / ratio
            if left <= lx < right and top <= ly < bottom:
                inside += 1
            else:
                outside.append((ix, iy))
    return outside, inside


def test_layer_list_draws_background_and_opened_project(tmp_path):
    """The default Background row is drawn, and every layer of an opened project is too."""
    from PIL import Image
    from PyQt6.QtWidgets import QApplication

    from pixedit.factory import raster_layer, redaction_layer, text_layer
    from pixedit.io import save_project
    from pixedit.model import Document, TextInfo
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    app.processEvents()
    layers = window.layers.list
    assert layers.topLevelItemCount() == 1
    background = layers.topLevelItem(0).child(0)
    assert background.text(0).startswith("Background")
    assert layers.visualItemRect(background).height() > 0

    document = Document(
        40,
        30,
        [
            raster_layer("Photo", Image.new("RGBA", (40, 30), (10, 20, 30, 255))),
            text_layer("Caption", TextInfo("Hi", None, "Default", 18, (255, 0, 0, 255)), 2, 2),
            redaction_layer("Blur", Image.new("L", (40, 30), 0), "blur", 8),
        ],
    )
    path = tmp_path / "stack.pix"
    save_project(document, str(path))
    window.open_path(str(path))
    app.processEvents()
    group = layers.topLevelItem(0)
    names = []
    for index in range(group.childCount()):
        item = group.child(index)
        assert layers.visualItemRect(item).height() > 0
        names.append(item.text(0).split()[0])
    assert names == ["Blur", "Caption", "Photo"]
    window.session.dirty = False
    window.close()
    app.processEvents()
