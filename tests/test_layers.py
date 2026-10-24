"""Layer stack commands: order, opacity, duplicate, delete, merge, undo."""

from __future__ import annotations

from PIL import Image

from pixedit.composite import composite
from pixedit.factory import raster_layer
from pixedit.model import new_document
from pixedit.session import Session
from pixedit.ui.panels import stack_move


def test_stack_ops_and_undo():
    session = Session(new_document(8, 8, "white"))
    assert len(session.document.layers) == 1
    session.add_blank_layer()
    assert len(session.document.layers) == 2
    session.document.layers[1].name = "Paint"
    session.rename_active("Sketch")
    assert session.active_layer().name == "Sketch"
    session.set_opacity(session.active_layer(), 0.4)
    assert session.active_layer().opacity == 0.4
    session.set_blend(session.active_layer(), "screen")
    assert session.active_layer().blend_mode == "screen"
    session.set_visibility(session.active_layer(), False)
    assert session.active_layer().visible is False
    session.duplicate_active()
    assert len(session.document.layers) == 3
    assert session.active_layer().name.endswith("copy")
    session.move_active(-1)
    session.delete_active()
    assert len(session.document.layers) == 2
    session.undo()
    assert len(session.document.layers) == 3
    session.redo()
    assert len(session.document.layers) == 2


def test_move_layer_can_jump_and_undo():
    session = Session(new_document(4, 4, "white"))
    session.document.layers[0].name = "Back"
    session.add_layer(raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255))))
    session.add_layer(raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255))))
    session.move_layer(2, 0)
    assert [layer.name for layer in session.document.layers] == ["Front", "Back", "Mid"]
    assert session.layer_index == 0
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Back", "Mid", "Front"]
    session.redo()
    assert [layer.name for layer in session.document.layers] == ["Front", "Back", "Mid"]


def test_stack_move_reads_a_front_first_list():
    # Unchanged list: front, mid, back.
    assert stack_move(["a", "b", "c"], ["c", "b", "a"]) is None
    # Drag the front layer to the back of the list.
    assert stack_move(["a", "b", "c"], ["b", "a", "c"]) == (2, 0)
    # Drag the back layer to the front of the list.
    assert stack_move(["a", "b", "c"], ["a", "c", "b"]) == (0, 2)
    # Reversing the whole stack is not a single drag.
    assert stack_move(["a", "b", "c"], ["a", "b", "c"]) is None


def test_merge_down_bakes_upper_layer():
    session = Session(new_document(4, 4, "white"))
    red = raster_layer("Red", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session.add_layer(red)
    session.layer_index = 1
    session.merge_down_active()
    assert len(session.document.layers) == 1
    pixel = session.document.layers[0].image.getpixel((1, 1))
    assert pixel[0] > 250
    assert pixel[1] < 5
    assert pixel[3] == 255
    session.undo()
    assert len(session.document.layers) == 2
    assert composite(session.document).getpixel((0, 0))[0] > 250


def test_layer_list_drop_reorders_the_document():
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.ui.panels import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(4, 4, "white"))
    session.document.layers[0].name = "Back"
    session.add_layer(raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255))))
    session.add_layer(raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255))))
    panel = LayerPanel(session, lambda: None)
    panel.refresh()
    icons = {button.accessibleName(): button for button in panel.findChildren(QToolButton)}
    assert {"New", "Copy", "Delete", "Up", "Down", "Visibility", "Group", "Merge down"} <= set(icons)
    for name, button in icons.items():
        assert not button.icon().isNull(), name
        if name == "Visibility":
            assert "selected layer" in button.toolTip()
        elif name == "Group":
            assert "own canvas" in button.toolTip()
        elif name == "Merge down":
            assert "below" in button.toolTip()
            assert "Flatten" in button.toolTip()
            assert "Ctrl+Shift+E" in button.toolTip()
        else:
            assert button.toolTip() == name
    # Visual order is Front, Mid, Back. Drop Front at the bottom of the group.
    panel.list.blockSignals(True)
    group = panel.list.topLevelItem(0)
    moved = group.takeChild(0)
    group.addChild(moved)
    panel.list.blockSignals(False)
    panel.list.drag_finished.emit()
    app.processEvents()
    assert [layer.name for layer in session.document.layers] == ["Front", "Back", "Mid"]
    assert session.layer_index == 0
    group = panel.list.topLevelItem(0)
    shown = [group.child(row).text(0).split()[0] for row in range(group.childCount())]
    assert shown == ["Mid", "Back", "Front"]
    panel.close()


def test_text_head_collapses_newlines_and_ellipsizes():
    from pixedit.ui.thumbs import text_head

    assert text_head("Hello\nthere  friend") == "Hello there friend"
    assert text_head("  spaced   out  ") == "spaced out"
    assert text_head("") == ""
    assert text_head("   \n  ") == ""
    short = "Hello, world"
    assert text_head(short) == short
    assert "…" not in text_head(short)
    longer = "Hello, world this is a long caption"
    assert text_head(longer, limit=22) == "Hello, world this is…"
    packed = "abcdefghijklmnopqrstuv"  # exactly the limit, then more
    assert text_head(packed + "wxyz", limit=22) == "abcdefghijklmnopqrstu…"
    assert len(text_head(packed + "wxyz", limit=22)) == 22
    assert text_head(longer[:22], limit=22) == longer[:22]


def test_layer_thumb_fits_ink_masks_and_empty():
    import numpy as np

    from pixedit.factory import redaction_layer
    from pixedit.ui.thumbs import CHECKER_DARK, CHECKER_LIGHT, MASK_TINT, layer_thumb

    mark = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    mark.putpixel((60, 60), (255, 0, 0, 255))
    filled = layer_thumb(raster_layer("Dot", mark), 32)
    assert filled.size == (32, 32)
    assert filled.getpixel((0, 0))[:3] == (255, 0, 0)
    assert filled.getpixel((31, 31))[:3] == (255, 0, 0)

    wide = Image.new("RGBA", (20, 10), (255, 255, 255, 255))
    letterboxed = layer_thumb(raster_layer("White", wide), 32)
    assert letterboxed.getpixel((16, 16))[:3] == (255, 255, 255)
    assert letterboxed.getpixel((0, 0))[:3] == CHECKER_LIGHT[:3]

    faint = raster_layer("Faint", Image.new("RGBA", (4, 4), (0, 80, 200, 255)))
    faint.opacity = 0.2
    assert layer_thumb(faint, 32).getpixel((8, 8))[:3] == (0, 80, 200)

    big = Image.new("RGBA", (180, 90), (0, 80, 200, 255))
    assert layer_thumb(raster_layer("Photo", big), 32).getpixel((16, 16))[:3] == (0, 80, 200)

    empty = layer_thumb(raster_layer("Empty", Image.new("RGBA", (1, 1), (0, 0, 0, 0))), 32)
    assert empty.getpixel((0, 0)) == CHECKER_LIGHT
    assert empty.getpixel((4, 0)) == CHECKER_DARK

    band = Image.new("L", (20, 20), 0)
    band.paste(255, (0, 8, 20, 12))
    mask_thumb = layer_thumb(redaction_layer("Band", band, "blur", 8), 32)
    assert mask_thumb.getpixel((16, 16))[:3] == MASK_TINT
    assert mask_thumb.getpixel((0, 0))[:3] == CHECKER_LIGHT[:3]

    soft = Image.new("L", (8, 8), 128)
    feathered = layer_thumb(redaction_layer("Soft", soft, "dim", 10), 32)
    alpha = 128 / 255.0
    expected = tuple(int(round(tint * alpha + light * (1.0 - alpha))) for tint, light in zip(MASK_TINT, CHECKER_LIGHT))
    assert feathered.getpixel((16, 16))[:3] == expected
    assert int(np.asarray(feathered)[..., 3].min()) == 255


def test_layer_rows_show_thumbs_and_text_heads():
    from PyQt6.QtWidgets import QApplication

    from pixedit.factory import redaction_layer, text_layer
    from pixedit.model import TextInfo
    from pixedit.ui.panels import LayerPanel
    from pixedit.ui.thumbs import MASK_TINT

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(16, 16, "white"))
    session.document.layers[0].name = "Photo"
    session.document.layers[0].opacity = 0.4
    words = "Hello\nthere friend, this caption is longer than the row"
    session.add_layer(text_layer("Caption", TextInfo(words, None, "Default", 18, (255, 0, 0, 255)), 0, 0))
    mask = Image.new("L", (16, 8), 0)
    mask.paste(255, (0, 2, 16, 6))
    session.add_layer(redaction_layer("Blur", mask, "blur", 8))
    panel = LayerPanel(session, lambda: None)
    panel.resize(280, 420)
    panel.show()
    panel.refresh()
    app.processEvents()

    # Front of the stack is the first row: Blur, Caption, Photo.
    group = panel.list.topLevelItem(0)
    blur, caption, photo = (group.child(row) for row in range(3))
    assert photo.text(0).splitlines() == ["Photo", "40%  normal"]
    assert not photo.icon(0).isNull()
    assert _icon_has_color(photo.icon(0), (255, 255, 255))

    assert caption.text(0).splitlines()[0] == "Caption"
    assert caption.text(0).splitlines()[1].startswith("“Hello there friend")
    assert caption.text(0).splitlines()[1].endswith("”  100%")
    assert "…" in caption.text(0)
    assert caption.toolTip(0) == words
    assert not caption.icon(0).isNull()
    assert _icon_has_color(caption.icon(0), (0xD5, 0xE4, 0xEE), tolerance=36)
    assert not _icon_has_color(photo.icon(0), (0xD5, 0xE4, 0xEE), tolerance=20)

    assert blur.text(0).splitlines()[0] == "  Blur"
    assert "on Caption" in blur.text(0).splitlines()[1]
    assert not blur.icon(0).isNull()
    assert _icon_has_color(blur.icon(0), MASK_TINT)
    assert panel.list.visualItemRect(caption).height() >= panel.list.iconSize().height()
    # A long caption stays inside the column instead of opening a sideways scrollbar.
    assert panel.list.horizontalScrollBar().maximum() == 0
    panel.close()


def _icon_has_color(icon, rgb: tuple[int, int, int], tolerance: int = 8) -> bool:
    """True when the row icon contains a pixel near ``rgb``.

    The pixmap is requested larger than the slot so a thin stroke still
    lands on a sample. Format_RGBA8888 is byte order R, G, B, A.
    """
    import numpy as np
    from PyQt6.QtGui import QImage

    pixmap = icon.pixmap(48, 48)
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    pointer = image.bits()
    pointer.setsize(image.sizeInBytes())
    width = image.width()
    row = image.bytesPerLine() // 4
    pixels = np.frombuffer(pointer, np.uint8).reshape((image.height(), row, 4))[:, :width, :3]
    target = np.array(rgb, np.int16)
    return bool(np.any(np.max(np.abs(pixels.astype(np.int16) - target), axis=-1) <= tolerance))


def test_visibility_toggle_flips_every_selected_layer():
    session = Session(new_document(8, 8, "white"))
    session.document.layers[0].name = "Back"
    mid = raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255)))
    front = raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session.add_layer(mid)
    session.add_layer(front)
    session.set_visibility(mid, False)
    session.select_layers([front.id, mid.id], active_id=front.id)
    session.toggle_selected_visibility()
    assert front.visible is False
    assert mid.visible is True
    assert session.document.layers[0].visible is True
    assert session.history.undo_stack[-1].name == "Toggle visibility"
    session.undo()
    assert front.visible is True
    assert mid.visible is False
    session.redo()
    assert front.visible is False
    assert mid.visible is True
    # A second toggle restores the mix, still as one step.
    steps = len(session.history.undo_stack)
    session.toggle_selected_visibility()
    assert front.visible is True
    assert mid.visible is False
    assert len(session.history.undo_stack) == steps + 1


def test_visibility_button_toggles_the_selection_and_refreshes_the_checks():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.ui.panels import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    session.document.layers[0].name = "Back"
    mid = raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255)))
    front = raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session.add_layer(mid)
    session.add_layer(front)
    panel = LayerPanel(session, lambda: None)
    panel.resize(240, 420)
    panel.show()
    panel.refresh()
    app.processEvents()
    session.select_layers([front.id, session.document.layers[0].id], active_id=front.id)
    panel.refresh()
    button = next(item for item in panel.findChildren(QToolButton) if item.accessibleName() == "Visibility")
    assert button.geometry().right() < panel.width()
    button.click()
    app.processEvents()
    assert front.visible is False
    assert session.document.layers[0].visible is False
    assert mid.visible is True
    panel.refresh()
    group = panel.list.topLevelItem(0)
    checks = {
        group.child(row).text(0).splitlines()[0]: group.child(row).checkState(0)
        for row in range(group.childCount())
    }
    assert checks["Front"] == Qt.CheckState.Unchecked
    assert checks["Back"] == Qt.CheckState.Unchecked
    assert checks["Mid"] == Qt.CheckState.Checked
    button.click()
    assert front.visible is True
    assert session.document.layers[0].visible is True
    panel.close()


def test_layer_index_selects_one_and_edits_keep_a_multi_selection():
    session = Session(new_document(8, 8, "white"))
    left = raster_layer("Left", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    right = raster_layer("Right", Image.new("RGBA", (4, 4), (0, 255, 0, 255)))
    session.add_layer(left)
    session.add_layer(right)
    session.select_layers([left.id, right.id], active_id=left.id)
    session.set_opacity(left, 0.4)
    assert set(session.selected_ids()) == {left.id, right.id}
    assert right.opacity == 1
    session.layer_index = session.document.index_of(right.id)
    assert session.selected_ids() == [right.id]

    session.select_layers([left.id, right.id], active_id=right.id)
    session.delete_active()
    assert all(layer.id != right.id for layer in session.document.layers)
    assert session.selected_ids() == [left.id]


def test_layer_list_selects_several_rows_and_keeps_the_active_one():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.panels import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    session.document.layers[0].name = "Back"
    mid = raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255)))
    mid.opacity = 0.5
    front = raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    front.opacity = 0.9
    session.add_layer(mid)
    session.add_layer(front)
    panel = LayerPanel(session, lambda: None)
    panel.refresh()
    # Visual order is Front, Mid, Back. Mid becomes the active layer.
    group = panel.list.topLevelItem(0)
    front_row = group.child(0)
    mid_row = group.child(1)
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(mid_row)
    front_row.setSelected(True)
    mid_row.setSelected(True)
    panel.list.blockSignals(False)
    panel._on_selection()
    assert session.active_layer().id == mid.id
    assert set(session.selected_ids()) == {front.id, mid.id}
    assert panel.opacity.value() == 50
    panel.refresh()
    group = panel.list.topLevelItem(0)
    selected = {
        group.child(row).data(0, Qt.ItemDataRole.UserRole)
        for row in range(group.childCount())
        if group.child(row).isSelected()
    }
    assert selected == {front.id, mid.id}
    assert panel.list.currentItem().data(0, Qt.ItemDataRole.UserRole) == mid.id
    session.set_opacity(session.active_layer(), 0.25)
    assert mid.opacity == 0.25
    assert front.opacity == 0.9
    assert set(session.selected_ids()) == {front.id, mid.id}
    panel.close()
    app.processEvents()


def test_brush_edits_only_the_active_layer_of_a_selection():
    from pixedit.tools.base import PointerEvent
    from pixedit.tools.builtins import BrushTool

    session = Session(new_document(48, 48, "white"))
    other = raster_layer("Other", Image.new("RGBA", (8, 8), (0, 0, 255, 255)), 30, 30)
    active = raster_layer("Active", Image.new("RGBA", (8, 8), (0, 255, 0, 255)), 4, 4)
    session.add_layer(other)
    session.add_layer(active)
    session.select_layers([other.id, active.id], active_id=active.id)
    session.foreground = (255, 0, 0, 255)
    session.options["brush_size"] = 4
    session.options["brush_hardness"] = 100
    before = other.image.tobytes()
    tool = BrushTool()
    tool.pointer_down(session, PointerEvent(6, 6))
    tool.pointer_up(session, PointerEvent(6, 6))
    assert other.image.tobytes() == before
    assert active.image.getpixel((2, 2))[0] > 200
    assert session.document.layers[0].image.getpixel((6, 6))[:3] == (255, 255, 255)


def test_groups_keep_separate_canvases_and_selections():
    session = Session(new_document(8, 8, "white"))
    first = session.document.active_group_id
    session.add_group()
    second = session.document.active_group_id
    assert second != first
    assert session.document.group.name == "Group 2"
    session.document.layers[0].image.putpixel((1, 1), (255, 0, 0, 255))
    session.select_all()
    session.activate_group(first)
    assert session.document.selection is None
    assert session.document.layers[0].image.getpixel((1, 1))[:3] == (255, 255, 255)
    session.crop_to(0, 0, 3, 3)
    assert (session.document.width, session.document.height) == (3, 3)
    session.activate_group(second)
    assert (session.document.width, session.document.height) == (8, 8)
    assert session.document.layers[0].image.getpixel((1, 1))[:3] == (255, 0, 0)
    assert session.document.selection is not None


def test_undo_of_another_page_shows_that_page():
    session = Session(new_document(8, 8, "white"))
    session.add_group()
    second = session.document.active_group_id
    first = session.document.groups[0].id
    session.activate_group(first)
    session.set_opacity(session.active_layer(), 0.25)
    session.activate_group(second)
    assert session.document.active_group_id == second
    session.undo()
    assert session.document.active_group_id == first
    assert session.active_layer().opacity == 1


def test_move_layer_to_another_group_and_refuse_a_lone_mask():
    from pixedit.factory import redaction_layer

    session = Session(new_document(8, 8, "white"))
    first = session.document.groups[0].id
    session.add_group()
    second = session.document.groups[1].id
    session.activate_group(first)
    photo = raster_layer("Photo", Image.new("RGBA", (4, 4), (0, 0, 255, 255)))
    session.add_layer(photo)
    mask = redaction_layer("Mask", Image.new("L", (8, 8), 0), "pixelate", 4, target_id=photo.id)
    session.add_layer(mask)
    assert session.move_layers_to_group([mask.id], second, 0) is False
    assert any(layer.id == mask.id for layer in session.document.groups[0].layers)
    assert session.move_layers_to_group([photo.id], second, 1) is True
    assert session.document.active_group_id == second
    names = [layer.name for layer in session.document.layers]
    assert "Photo" in names and "Mask" in names
    assert all(layer.name not in {"Photo", "Mask"} for layer in session.document.groups[0].layers)
    session.undo()
    assert session.document.active_group_id == first
    assert any(layer.name == "Photo" for layer in session.document.layers)


def test_delete_group_keeps_the_last_page_and_reorder_sets_page_order():
    session = Session(new_document(4, 4, "white"))
    session.delete_active_group()
    assert len(session.document.groups) == 1
    session.add_group()
    session.rename_group(session.document.groups[0].id, "Page 1")
    session.rename_group(session.document.groups[1].id, "Page 2")
    ordered = [session.document.groups[1].id, session.document.groups[0].id]
    session.reorder_groups(ordered)
    assert [group.name for group in session.document.groups] == ["Page 2", "Page 1"]
    session.undo()
    assert [group.name for group in session.document.groups] == ["Page 1", "Page 2"]
    session.activate_group(session.document.groups[1].id)
    session.delete_active_group()
    assert [group.name for group in session.document.groups] == ["Page 1"]
    session.undo()
    assert [group.name for group in session.document.groups] == ["Page 1", "Page 2"]


def test_clicking_a_group_layer_shows_that_page():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.panels import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    session.add_group()
    first = session.document.groups[0]
    second = session.document.groups[1]
    session.activate_group(first.id)
    panel = LayerPanel(session, lambda: None)
    panel.refresh()
    row = panel.list.topLevelItem(1).child(0)
    panel.list.setCurrentItem(row)
    panel._on_selection()
    assert session.document.active_group_id == second.id
    panel.close()
    app.processEvents()


def test_canvas_restores_zoom_for_each_group():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.canvas import CanvasView

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(40, 30, "white"))
    canvas = CanvasView(session, lambda: None, lambda *_args: None)
    canvas.resize(400, 300)
    canvas.show()
    app.processEvents()
    first = session.document.active_group_id
    session.add_group()
    canvas.invalidate()
    app.processEvents()
    canvas.zoom = 4
    canvas.offset_x = 15
    canvas.offset_y = 18
    session.activate_group(first)
    canvas.invalidate()
    app.processEvents()
    assert canvas.zoom != 4
    session.activate_group(session.document.groups[1].id)
    canvas.invalidate()
    app.processEvents()
    assert canvas.zoom == 4
    assert canvas.offset_x == 15
    assert canvas.offset_y == 18
    canvas.close()
    app.processEvents()
