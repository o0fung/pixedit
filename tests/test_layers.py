"""Layer stack commands: order, opacity, duplicate, delete, merge, undo."""

from __future__ import annotations

from PIL import Image

from pixedit.composite import composite
from pixedit.factory import raster_layer
from pixedit.model import new_document
from pixedit.session import Session
from pixedit.ui.layers_panel import stack_move


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


def test_merge_selected_layers_leaves_the_one_between_them():
    """Several selected layers bake together. A layer between them stays put."""
    session = Session(new_document(4, 4, "white"))
    back = session.document.layers[0]
    back.name = "Back"
    back.image = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    session.add_layer(raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255))))
    front_image = Image.new("RGBA", (4, 4), (0, 0, 255, 255))
    front_image.putpixel((0, 0), (0, 0, 0, 0))
    front = raster_layer("Front", front_image)
    session.add_layer(front)
    session.select_layers([back.id, front.id], active_id=front.id)
    session.merge_down_active()
    assert [layer.name for layer in session.document.layers] == ["Mid", "Front"]
    merged = session.document.layers[1]
    assert merged.id == front.id
    # The clear corner of Front still shows Back. Mid was not part of the plate.
    assert merged.image.getpixel((0, 0))[:3] == (255, 0, 0)
    assert merged.image.getpixel((1, 1))[:3] == (0, 0, 255)
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (0, 255, 0)
    assert session.history.undo_stack[-1].name == "Merge layers"
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Back", "Mid", "Front"]


def test_selected_mask_merges_into_its_picture_not_the_layer_below():
    from pixedit.factory import redaction_layer

    session = Session(new_document(4, 4, "white"))
    below = session.document.layers[0]
    below.name = "Below"
    below.image = Image.new("RGBA", (4, 4), (0, 0, 255, 255))
    picture = raster_layer("Picture", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session.add_layer(picture)
    cover = redaction_layer(
        "Cover",
        Image.new("L", (4, 4), 255),
        "solid",
        1,
        color=(0, 255, 0, 255),
        target_id=picture.id,
    )
    session.add_layer(cover)
    session.select_layers([picture.id, cover.id], active_id=cover.id)
    session.merge_down_active()
    assert [layer.name for layer in session.document.layers] == ["Below", "Cover"]
    assert session.document.layers[0].image.getpixel((0, 0))[:3] == (0, 0, 255)
    assert session.document.layers[1].image.getpixel((0, 0))[:3] == (0, 255, 0)
    assert session.history.undo_stack[-1].name == "Merge layers"


def test_merge_of_only_masks_asks_for_layers():
    from pixedit.factory import redaction_layer

    session = Session(new_document(4, 4, "white"))
    picture = session.document.layers[0]
    one = redaction_layer("One", Image.new("L", (4, 4), 255), "solid", 1, color=(0, 0, 0, 255), target_id=picture.id)
    two = redaction_layer("Two", Image.new("L", (4, 4), 255), "solid", 1, color=(0, 0, 0, 255), target_id=picture.id)
    session.add_layer(one)
    session.add_layer(two)
    session.select_layers([one.id, two.id], active_id=two.id)
    messages = []
    session.set_message_handler(messages.append)
    before = [layer.id for layer in session.document.layers]
    session.merge_down_active()
    assert [layer.id for layer in session.document.layers] == before
    assert messages == ["Select the layers to merge."]


def test_layer_list_drop_reorders_the_document():
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(4, 4, "white"))
    session.document.layers[0].name = "Back"
    session.add_layer(raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255))))
    session.add_layer(raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255))))
    panel = LayerPanel(session)
    panel.refresh()
    icons = {button.accessibleName(): button for button in panel.findChildren(QToolButton)}
    assert {"Merge down", "Page", "New tag", "Tag pages"} <= set(icons)
    assert "Visibility" not in icons and "New" not in icons and "Up" not in icons and "Down" not in icons
    assert panel._groups_empty.text() == "No tags yet."
    assert panel._groups_empty.isVisibleTo(panel)
    assert not panel.groups.isVisibleTo(panel)
    for name, button in icons.items():
        assert not button.icon().isNull(), name
        if name == "Page":
            assert "own canvas" in button.toolTip()
        elif name == "New tag":
            assert "tag" in button.toolTip().lower()
        elif name == "Tag pages":
            assert "tag" in button.toolTip().lower()
        elif name == "Merge down":
            assert "below" in button.toolTip()
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


def test_layer_thumb_outlines_dark_ink_and_leaves_light_ink():
    import numpy as np

    from pixedit.ui.thumbs import CHECKER_LIGHT, KEYLINE, layer_thumb

    black = np.asarray(layer_thumb(raster_layer("Black", Image.new("RGBA", (16, 8), (0, 0, 0, 255))), 32))
    # The bar is letterboxed. Its center stays black, and the pixel just
    # outside it is the rim. The far corner is still the checker.
    assert tuple(black[16, 16, :3]) == (0, 0, 0)
    assert tuple(black[7, 16, :3]) == KEYLINE[:3]
    assert tuple(black[0, 0, :3]) == CHECKER_LIGHT[:3]

    slab = layer_thumb(raster_layer("Slab", Image.new("RGBA", (8, 8), (0, 0, 0, 255))), 32)
    assert slab.getpixel((16, 16))[:3] == (0, 0, 0)
    assert slab.getpixel((0, 0))[:3] == KEYLINE[:3]

    white = np.asarray(layer_thumb(raster_layer("White", Image.new("RGBA", (20, 10), (255, 255, 255, 255))), 32))
    assert not np.any(np.all(white[:, :, :3] == np.array(KEYLINE[:3]), axis=-1))
    assert tuple(white[0, 0, :3]) == CHECKER_LIGHT[:3]
    assert tuple(white[16, 16, :3]) == (255, 255, 255)

    empty = layer_thumb(raster_layer("Empty", Image.new("RGBA", (1, 1), (0, 0, 0, 0))), 32)
    assert empty.getpixel((0, 0)) == CHECKER_LIGHT

    red = layer_thumb(raster_layer("Red", Image.new("RGBA", (8, 8), (255, 0, 0, 255))), 32)
    assert red.getpixel((0, 0))[:3] == (255, 0, 0)
    assert red.getpixel((16, 16))[:3] == (255, 0, 0)

    blue = layer_thumb(raster_layer("Blue", Image.new("RGBA", (4, 4), (0, 80, 200, 255))), 32)
    assert blue.getpixel((16, 16))[:3] == (0, 80, 200)
    assert blue.getpixel((0, 0))[:3] == KEYLINE[:3]

    # A white pixel beside black keeps its color. The rim does not eat it.
    mixed = Image.new("RGBA", (8, 8), (0, 0, 0, 255))
    mixed.paste((255, 255, 255, 255), (4, 0, 8, 8))
    side_by_side = np.asarray(layer_thumb(raster_layer("Mixed", mixed), 32))
    assert tuple(side_by_side[16, 8, :3]) == (0, 0, 0)
    assert tuple(side_by_side[16, 24, :3]) == (255, 255, 255)


def test_layer_rows_show_thumbs_and_text_heads():
    from PyQt6.QtWidgets import QApplication

    from pixedit.factory import redaction_layer, text_layer
    from pixedit.model import TextInfo
    from pixedit.ui.layers_panel import _ITEM_META, _ITEM_NAME, LayerPanel
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
    panel = LayerPanel(session)
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
    assert group.data(0, _ITEM_NAME) == "Page 1"
    assert group.data(0, _ITEM_META) == "16 × 16"
    assert photo.data(0, _ITEM_NAME) == "Photo"
    assert photo.data(0, _ITEM_META) == "40%  normal"
    assert blur.data(0, _ITEM_NAME) == "  Blur"
    assert blur.data(0, _ITEM_META) == blur.text(0).splitlines()[1]
    assert not blur.icon(0).isNull()
    assert _icon_has_color(blur.icon(0), MASK_TINT)
    assert panel.list.visualItemRect(caption).height() >= panel.list.iconSize().height()
    # A long caption stays inside the column instead of opening a sideways scrollbar.
    assert panel.list.horizontalScrollBar().maximum() == 0
    panel.close()


def test_row_paint_keeps_the_name_brighter_than_the_caption():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import _CAPTION_COLOR, _CAPTION_SELECTED, _NAME_COLOR, LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(16, 16, "white"))
    session.document.layers[0].name = "Photo"
    session.document.layers[0].opacity = 0.4
    panel = LayerPanel(session)
    panel.resize(360, 240)
    panel.show()
    panel.refresh()
    app.processEvents()
    group = panel.list.topLevelItem(0)
    photo = group.child(0)

    plain = _row_pixels(panel, photo, selected=False)
    name_y, caption_y = _color_bands(plain, _NAME_COLOR, _CAPTION_COLOR)
    assert name_y < caption_y

    chosen = _row_pixels(panel, photo, selected=True)
    _name_y, selected_y = _color_bands(chosen, _NAME_COLOR, _CAPTION_SELECTED)
    assert _name_y < selected_y

    page = _row_pixels(panel, group, selected=False)
    name_x, size_x = _color_columns(page, _NAME_COLOR, _CAPTION_COLOR)
    assert name_x < size_x
    panel.close()


def test_page_tags_each_get_a_line_and_the_buttons_share_a_row():
    import numpy as np
    from PyQt6.QtGui import QFontMetrics
    from PyQt6.QtWidgets import QApplication, QLabel, QToolButton

    from pixedit.ui.icons import icon_named
    from pixedit.ui.layers_panel import (
        _CAPTION_COLOR,
        _ITEM_META,
        _ITEM_TAGS,
        _NAME_COLOR,
        _caption_font,
        LayerPanel,
        branch_line_rect,
    )

    assert not icon_named("tag-new").isNull()

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(16, 16, "white"))
    session.add_shared_group()
    session.add_shared_group()
    session.add_shared_group()
    assert [group.name for group in session.document.shared_groups] == ["Tag 1", "Tag 2", "Tag 3"]
    session.add_group()
    session.document.group.shared_ids = []
    panel = LayerPanel(session)
    panel.resize(320, 640)
    panel.show()
    panel.refresh()
    app.processEvents()

    sections = [label.text() for label in panel.findChildren(QLabel) if label.objectName() == "section"]
    assert sections == ["Pages", "Tags"]
    buttons = {button.accessibleName(): button for button in panel.findChildren(QToolButton)}
    row = buttons["Page"].y()
    assert buttons["New tag"].y() == row
    assert buttons["Tag pages"].y() == row
    assert buttons["Merge down"].y() == row
    assert buttons["Page"].x() < buttons["New tag"].x() < buttons["Tag pages"].x() < buttons["Merge down"].x()

    tagged = panel.list.topLevelItem(0)
    bare = panel.list.topLevelItem(1)
    assert tagged.data(0, _ITEM_META) == "16 × 16"
    assert list(tagged.data(0, _ITEM_TAGS)) == ["Tag 1", "Tag 2", "Tag 3"]
    assert tagged.toolTip(0) == "Tag 1\nTag 2\nTag 3"
    assert list(bare.data(0, _ITEM_TAGS) or []) == []
    assert bare.data(0, _ITEM_META) == "16 × 16"

    caption_h = QFontMetrics(_caption_font(panel.font())).height()
    delta = panel.list.visualItemRect(tagged).height() - panel.list.visualItemRect(bare).height()
    # Three tag names, plus the blank line the fold arrow sits on.
    assert abs(delta - 4 * caption_h) <= caption_h
    tagged_row = panel.list.visualItemRect(tagged)
    branch = branch_line_rect(tagged_row, caption_h)
    assert branch.bottom() == tagged_row.bottom()
    assert branch.top() > tagged_row.top()
    layer = tagged.child(0)
    assert panel.list.visualItemRect(layer).top() > panel.list.visualItemRect(tagged).bottom()

    page = _row_pixels(panel, tagged, selected=False, height=280)
    name = _nearest_mask(page, _NAME_COLOR, _CAPTION_COLOR)
    caption = _nearest_mask(page, _CAPTION_COLOR, _NAME_COLOR)
    name_rows = np.nonzero(name.any(axis=1))[0]
    bands = _pixel_bands(caption)
    assert len(bands) == 4
    size_top, size_bottom = bands[0]
    assert size_top <= int(name_rows.mean()) <= size_bottom
    size_cols = caption[size_top : size_bottom + 1].any(axis=0).nonzero()[0]
    name_cols = name.any(axis=0).nonzero()[0]
    assert float(name_cols.mean()) < float(size_cols.mean())
    previous_bottom = int(name_rows.max())
    for top, bottom in bands[1:]:
        assert top > previous_bottom
        previous_bottom = bottom
    panel.close()


def _row_pixels(panel, item, selected: bool, height: int = 72, point_size: int = 15):
    """RGB pixels of one row, cropped to the text the delegate paints."""
    import numpy as np
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem

    image = QImage(420, height, QImage.Format.Format_RGB32)
    image.fill(QColor("#1a1f26"))
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, image.width(), image.height())
    option.font = panel.font()
    option.font.setPointSize(point_size)
    option.widget = panel.list
    option.decorationSize = panel.list.iconSize()
    option.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Active
    if selected:
        option.state |= QStyle.StateFlag.State_Selected
    index = panel.list.indexFromItem(item, 0)
    painter = QPainter(image)
    panel.list.itemDelegate().paint(painter, option, index)
    painter.end()
    probe = QStyleOptionViewItem(option)
    panel.list.itemDelegate().initStyleOption(probe, index)
    text = panel.list.style().subElementRect(QStyle.SubElement.SE_ItemViewItemText, probe, panel.list)
    view = image.convertToFormat(QImage.Format.Format_RGBA8888)
    pointer = view.bits()
    pointer.setsize(view.sizeInBytes())
    row_stride = view.bytesPerLine() // 4
    pixels = np.frombuffer(pointer, np.uint8).reshape((view.height(), row_stride, 4))[:, : view.width(), :3]
    box = text.intersected(QRect(0, 0, view.width(), view.height()))
    return np.array(pixels[box.top() : box.bottom() + 1, box.left() : box.right() + 1], copy=True)


def _nearest_mask(pixels, color, other, tolerance: int = 36):
    """Pixels closer to ``color`` than to ``other``, and within ``tolerance``."""
    import numpy as np

    rgb = pixels.astype(np.int16)
    wanted = np.array((color.red(), color.green(), color.blue()), np.int16)
    rival = np.array((other.red(), other.green(), other.blue()), np.int16)
    dist = np.max(np.abs(rgb - wanted), axis=-1)
    rival_dist = np.max(np.abs(rgb - rival), axis=-1)
    return (dist <= tolerance) & (dist < rival_dist)


def _color_bands(pixels, name_color, caption_color) -> tuple[float, float]:
    name = _nearest_mask(pixels, name_color, caption_color)
    caption = _nearest_mask(pixels, caption_color, name_color)
    assert name.any(), "name color missing"
    assert caption.any(), "caption color missing"
    return float(np_rows(name).mean()), float(np_rows(caption).mean())


def _pixel_bands(mask) -> list[tuple[int, int]]:
    """Runs of set rows. A one-pixel hole stays in the run.

    Antialiasing can leave a speck just under a glyph. A real line break is
    several rows of background, so that speck does not become its own line.
    """
    import numpy as np

    rows = np.nonzero(mask.any(axis=1))[0]
    if len(rows) == 0:
        return []
    bands: list[tuple[int, int]] = []
    start = previous = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row > previous + 2:
            bands.append((start, previous))
            start = row
        previous = row
    bands.append((start, previous))
    return bands


def _color_columns(pixels, name_color, caption_color) -> tuple[float, float]:
    name = _nearest_mask(pixels, name_color, caption_color)
    caption = _nearest_mask(pixels, caption_color, name_color)
    assert name.any(), "name color missing"
    assert caption.any(), "caption color missing"
    return float(name.any(axis=0).nonzero()[0].mean()), float(caption.any(axis=0).nonzero()[0].mean())


def np_rows(mask):
    import numpy as np

    return np.nonzero(mask)[0]


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


def test_selected_visibility_follows_one_check_state():
    session = Session(new_document(8, 8, "white"))
    session.document.layers[0].name = "Back"
    mid = raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255)))
    front = raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session.add_layer(mid)
    session.add_layer(front)
    session.select_layers([front.id, mid.id], active_id=front.id)
    session.set_selected_visibility(False)
    assert front.visible is False
    assert mid.visible is False
    assert session.document.layers[0].visible is True
    assert session.history.undo_stack[-1].name == "Toggle visibility"
    session.undo()
    assert front.visible is True
    assert mid.visible is True
    session.redo()
    assert front.visible is False
    assert mid.visible is False


def test_checkbox_sets_every_selected_layer_and_leaves_the_rest():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    session.document.layers[0].name = "Back"
    mid = raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255)))
    front = raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session.add_layer(mid)
    session.add_layer(front)
    panel = LayerPanel(session)
    panel.resize(240, 420)
    panel.show()
    panel.refresh()
    app.processEvents()
    session.select_layers([front.id, session.document.layers[0].id], active_id=front.id)
    panel.refresh()
    group = panel.list.topLevelItem(0)
    assert group is not None
    front_row = next(group.child(row) for row in range(group.childCount()) if group.child(row).text(0).startswith("Front"))
    front_row.setCheckState(0, Qt.CheckState.Unchecked)
    app.processEvents()
    assert front.visible is False
    assert session.document.layers[0].visible is False
    assert mid.visible is True
    assert session.history.undo_stack[-1].name == "Toggle visibility"
    panel.refresh()
    group = panel.list.topLevelItem(0)
    assert group is not None
    checks = {
        group.child(row).text(0).splitlines()[0]: group.child(row).checkState(0)
        for row in range(group.childCount())
    }
    assert checks["Front"] == Qt.CheckState.Unchecked
    assert checks["Back"] == Qt.CheckState.Unchecked
    assert checks["Mid"] == Qt.CheckState.Checked
    session.undo()
    assert front.visible is True
    assert session.document.layers[0].visible is True

    # A row outside the selection changes only itself.
    session.select_layers([front.id, session.document.layers[0].id], active_id=front.id)
    panel.refresh()
    group = panel.list.topLevelItem(0)
    assert group is not None
    mid_row = next(group.child(row) for row in range(group.childCount()) if group.child(row).text(0).startswith("Mid"))
    mid_row.setCheckState(0, Qt.CheckState.Unchecked)
    app.processEvents()
    assert mid.visible is False
    assert front.visible is True
    assert session.document.layers[0].visible is True
    panel.close()


def test_checkbox_refresh_waits_until_set_data_returns():
    """A listener that rebuilds the list must not delete the row inside setData.

    The window refreshes on every session change. The checkbox emits that
    change from inside QTreeWidgetItem::setData, and clearing the list there
    crashes. The rebuild waits until the click has returned.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    layer = session.document.layers[0]
    layer.name = "Photo"
    panel = LayerPanel(session)
    session.add_listener(lambda kind: panel.refresh() if kind == "change" else None)
    panel.resize(240, 420)
    panel.show()
    panel.refresh()
    app.processEvents()
    group = panel.list.topLevelItem(0)
    assert group is not None
    row = group.child(0)
    assert row is not None
    row.setCheckState(0, Qt.CheckState.Unchecked)
    # setData has returned and the row it updated is still in the list.
    assert row.treeWidget() is panel.list
    assert layer.visible is False
    app.processEvents()
    group = panel.list.topLevelItem(0)
    assert group is not None
    rebuilt = group.child(0)
    assert rebuilt is not None
    assert rebuilt is not row
    assert rebuilt.text(0).startswith("Photo")
    assert rebuilt.checkState(0) == Qt.CheckState.Unchecked
    panel.close()
    app.processEvents()


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

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    session.document.layers[0].name = "Back"
    mid = raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255)))
    mid.opacity = 0.5
    front = raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    front.opacity = 0.9
    session.add_layer(mid)
    session.add_layer(front)
    panel = LayerPanel(session)
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
    from pixedit.tools.brush import BrushTool

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
    assert session.document.group.name == "Page 2"
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


def test_delete_groups_removes_selected_pages_in_one_undo():
    session = Session(new_document(4, 4, "white"))
    session.rename_group(session.document.groups[0].id, "Page 1")
    session.add_group()
    session.rename_group(session.document.active_group_id, "Page 2")
    session.add_group()
    session.rename_group(session.document.active_group_id, "Page 3")
    pages = session.document.groups
    first, second, third = pages[0].id, pages[1].id, pages[2].id
    session.activate_group(second)
    session.document.layers[0].name = "On page 2"
    # A missing id is skipped. Pages are removed in document order.
    depth = len(session.history.undo_stack)
    session.delete_groups(["missing", third, second])
    assert [group.name for group in session.document.groups] == ["Page 1"]
    assert session.document.active_group_id == first
    assert len(session.history.undo_stack) == depth + 1
    assert session.history.undo_stack[-1].name == "Delete pages"
    session.undo()
    assert [group.name for group in session.document.groups] == ["Page 1", "Page 2", "Page 3"]
    assert session.document.active_group_id == second
    assert session.document.layers[0].name == "On page 2"
    session.redo()
    assert [group.name for group in session.document.groups] == ["Page 1"]
    session.undo()
    session.delete_groups([])
    session.delete_groups([first])
    assert [group.name for group in session.document.groups] == ["Page 2", "Page 3"]
    assert session.history.undo_stack[-1].name == "Delete page"
    session.undo()


def test_delete_groups_refuses_to_remove_the_last_page():
    session = Session(new_document(4, 4, "white"))
    messages: list[str] = []
    session.set_message_handler(messages.append)
    only = session.document.groups[0].id
    session.delete_groups([only])
    assert len(session.document.groups) == 1
    assert messages == ["The last page stays. A document always has one page."]
    assert session.history.undo_stack == []
    session.add_group()
    session.add_group()
    messages.clear()
    depth = len(session.history.undo_stack)
    session.delete_groups([group.id for group in session.document.groups])
    assert len(session.document.groups) == 3
    assert len(session.history.undo_stack) == depth
    assert messages == ["The last page stays. A document always has one page."]


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


def test_place_group_block_keeps_the_selected_pages_together():
    from pixedit.ui.layers_panel import place_group_block

    model = ["A", "B", "C", "D"]
    # Only C moved to the end. B was also selected, so B travels with C.
    assert place_group_block(model, ["A", "B", "D", "C"], ["B", "C"], "C") == ["A", "D", "B", "C"]
    # The tree moved the whole block. The same pages still land together.
    assert place_group_block(model, ["A", "D", "B", "C"], ["B", "C"], "C") == ["A", "D", "B", "C"]
    assert place_group_block(model, ["D", "A", "B", "C"], ["B", "D"], "D") == ["B", "D", "A", "C"]


def test_folded_pages_stay_folded_and_drag_as_a_block():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(4, 4, "white"))
    names = ["A", "B", "C", "D"]
    session.rename_group(session.document.groups[0].id, names[0])
    for name in names[1:]:
        session.add_group()
        session.rename_group(session.document.active_group_id, name)
    panel = LayerPanel(session)
    panel.show()
    app.processEvents()
    panel.refresh()
    for index in range(panel.list.topLevelItemCount()):
        header = panel.list.topLevelItem(index)
        header.setExpanded(False)
        panel._on_group_folded(header)
    panel.refresh()
    assert all(not panel.list.topLevelItem(index).isExpanded() for index in range(4))

    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(panel.list.topLevelItem(2))
    panel.list.topLevelItem(1).setSelected(True)
    panel.list.topLevelItem(2).setSelected(True)
    panel.list.blockSignals(False)
    panel._on_selection()
    panel.refresh()
    selected = [
        panel.list.topLevelItem(index).text(0).split()[0]
        for index in range(4)
        if panel.list.topLevelItem(index).isSelected()
    ]
    assert selected == ["B", "C"]

    taken = panel.list.takeTopLevelItem(2)
    panel.list.addTopLevelItem(taken)
    panel._drag_groups = [session.document.groups[1].id, session.document.groups[2].id]
    panel._drag_current = session.document.groups[2].id
    panel._apply_pending_drop()
    assert [group.name for group in session.document.groups] == ["A", "D", "B", "C"]
    assert all(not panel.list.topLevelItem(index).isExpanded() for index in range(4))
    selected = [
        panel.list.topLevelItem(index).text(0).split()[0]
        for index in range(4)
        if panel.list.topLevelItem(index).isSelected()
    ]
    assert selected == ["B", "C"]
    session.undo()
    assert [group.name for group in session.document.groups] == ["A", "B", "C", "D"]
    panel.close()
    app.processEvents()


def test_delete_key_removes_a_selected_open_group():
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    def press(tree, key) -> None:
        tree.setFocus(Qt.FocusReason.MouseFocusReason)
        event = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(tree, event)

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    session.rename_group(session.document.groups[0].id, "One")
    first = session.document.groups[0].id
    session.add_group()
    session.rename_group(session.document.active_group_id, "Two")
    second = session.document.groups[1].id
    session.activate_group(first)
    session.add_blank_layer()
    panel = LayerPanel(session)
    panel.show()
    app.processEvents()
    panel.refresh()
    # An open group name stays selected. The layer under it does not.
    panel.list.setCurrentItem(panel.list.topLevelItem(0))
    app.processEvents()
    header = panel.list.topLevelItem(0)
    assert header is not None
    assert header.isSelected()
    assert header.child(0) is not None and not header.child(0).isSelected()
    assert panel._selected_groups == [first]
    assert session.document.active_group_id == first

    panel.list.setCurrentItem(panel.list.topLevelItem(1))
    app.processEvents()
    other = panel.list.topLevelItem(1)
    assert other is not None and other.isSelected()
    assert other.child(0) is not None and not other.child(0).isSelected()
    assert panel._selected_groups == [second]
    assert session.document.active_group_id == second

    press(panel.list, Qt.Key.Key_Delete)
    app.processEvents()
    assert [group.name for group in session.document.groups] == ["One"]
    session.undo()
    assert [group.name for group in session.document.groups] == ["One", "Two"]
    assert session.document.active_group_id == second

    # A layer row still deletes that layer and leaves both pages.
    panel.refresh()
    page = panel.list.topLevelItem(0)
    assert page is not None
    layer_row = page.child(0)
    assert layer_row is not None
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(layer_row)
    layer_row.setSelected(True)
    panel.list.blockSignals(False)
    panel._on_selection()
    assert panel._selected_groups == []
    layer_count = len(session.document.groups[0].layers)
    press(panel.list, Qt.Key.Key_Backspace)
    assert len(session.document.groups) == 2
    assert len(session.document.groups[0].layers) == layer_count - 1
    panel.close()
    app.processEvents()


def test_clicking_a_group_layer_shows_that_page():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    session.add_group()
    first = session.document.groups[0]
    second = session.document.groups[1]
    session.activate_group(first.id)
    panel = LayerPanel(session)
    panel.refresh()
    row = panel.list.topLevelItem(1).child(0)
    panel.list.setCurrentItem(row)
    panel._on_selection()
    assert session.document.active_group_id == second.id
    panel.close()
    app.processEvents()


def test_pix_filename_names_automatic_groups_and_keeps_a_typed_name():
    from pixedit.model import LayerGroup
    from pixedit.session.layers import apply_pix_group_names

    one = LayerGroup(id="a", name="Group 1", width=1, height=1)
    apply_pix_group_names([one], "/tmp/holiday.pix")
    assert one.name == "holiday"

    pages = [
        LayerGroup(id="a", name="Group 1", width=1, height=1),
        LayerGroup(id="b", name="Group 2", width=1, height=1),
        LayerGroup(id="c", name="Cover", width=1, height=1),
    ]
    apply_pix_group_names(pages, "album.pix")
    assert [group.name for group in pages] == ["album 1", "album 2", "Cover"]

    taken = [
        LayerGroup(id="a", name="Group 1", width=1, height=1),
        LayerGroup(id="b", name="holiday", width=1, height=1),
    ]
    apply_pix_group_names(taken, "holiday.pix")
    assert [group.name for group in taken] == ["holiday 2", "holiday"]

    bumped = [
        LayerGroup(id="a", name="Group 1", width=1, height=1),
        LayerGroup(id="b", name="Group 3", width=1, height=1),
        LayerGroup(id="c", name="notes 1", width=1, height=1),
    ]
    apply_pix_group_names(bumped, "notes.pix")
    assert [group.name for group in bumped] == ["notes 2", "notes 3", "notes 1"]


def test_open_pix_names_an_automatic_group_from_the_filename(tmp_path):
    from PyQt6.QtWidgets import QApplication

    from pixedit.io import save_project
    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    source = Session(new_document(4, 4, "white"))
    holiday = tmp_path / "holiday.pix"
    save_project(source.document, str(holiday))
    window = MainWindow()
    window.show()
    app.processEvents()
    window.open_path(str(holiday))
    app.processEvents()
    assert window.session.document.group.name == "holiday"
    window.session.add_group()
    assert window.session.document.group.name == "Page 2"

    source.document.groups[0].name = "Cover"
    cover = tmp_path / "cover.pix"
    save_project(source.document, str(cover))
    window.open_path(str(cover))
    app.processEvents()
    assert window.session.document.group.name == "Cover"

    extra = Session(new_document(4, 4, "white"))
    extra.add_group()
    notes = tmp_path / "notes.pix"
    save_project(extra.document, str(notes))
    window.open_path(str(holiday))
    app.processEvents()
    window._place_project(str(notes))
    app.processEvents()
    assert [group.name for group in window.session.document.groups] == ["holiday", "notes 1", "notes 2"]
    window.session.dirty = False
    window.close()
    app.processEvents()


def test_rename_dialog_edits_the_selected_group_or_layer(monkeypatch):
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(8, 8, "white"))
    panel = LayerPanel(session)
    panel.show()
    app.processEvents()
    panel.refresh()
    assert panel.list.expandsOnDoubleClick() is False

    header = panel.list.topLevelItem(0)
    assert header is not None
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(header)
    header.setSelected(True)
    panel.list.blockSignals(False)
    panel._on_selection()
    seen: dict[str, str] = {}

    def group_name(*_args, **kwargs):
        seen["title"] = kwargs.get("title", _args[1] if len(_args) > 1 else "")
        seen["text"] = kwargs.get("text", "")
        return "Cover", True

    monkeypatch.setattr("pixedit.ui.layers_panel.QInputDialog.getText", group_name)
    panel.rename_selection()
    assert seen["title"] == "Rename page"
    assert seen["text"] == "Page 1"
    assert session.document.group.name == "Cover"

    monkeypatch.setattr("pixedit.ui.layers_panel.QInputDialog.getText", lambda *_a, **_k: ("  ", True))
    panel.rename_selection()
    assert session.document.group.name == "Cover"
    monkeypatch.setattr("pixedit.ui.layers_panel.QInputDialog.getText", lambda *_a, **_k: ("Nope", False))
    panel.rename_selection()
    assert session.document.group.name == "Cover"

    panel.refresh()
    page = panel.list.topLevelItem(0)
    assert page is not None
    layer_row = page.child(0)
    assert layer_row is not None
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(layer_row)
    layer_row.setSelected(True)
    panel.list.blockSignals(False)
    panel._on_selection()
    assert panel._selected_groups == []

    def layer_name(*_args, **kwargs):
        seen["title"] = kwargs.get("title", _args[1] if len(_args) > 1 else "")
        return "Sketch", True

    monkeypatch.setattr("pixedit.ui.layers_panel.QInputDialog.getText", layer_name)
    panel.rename_selection()
    assert seen["title"] == "Rename layer"
    assert session.active_layer().name == "Sketch"
    assert session.document.group.name == "Cover"
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


def test_copy_and_paste_duplicates_the_active_layer():
    """Ctrl+C stores the layer. Ctrl+V inserts an independent copy above the current one."""
    from pixedit.factory import redaction_layer

    session = Session(new_document(8, 8, "white"))
    back = raster_layer("Back", Image.new("RGBA", (4, 4), (0, 0, 255, 255)), 0, 0)
    photo = raster_layer("Photo", Image.new("RGBA", (4, 4), (255, 0, 0, 255)), 1, 2)
    session.add_layer(back)
    session.add_layer(photo)
    session.select_layers([back.id, photo.id], active_id=photo.id)
    session.active_layer().opacity = 0.5
    session.active_layer().blend_mode = "multiply"
    steps = len(session.history.undo_stack)
    assert session.copy_active_layer()
    assert len(session.history.undo_stack) == steps
    # Paint after the copy stays on the original. The snapshot already left.
    photo.image.putpixel((0, 0), (0, 255, 0, 255))
    assert session.paste_copied_layer()
    pasted = session.active_layer()
    assert pasted.name == "Photo copy"
    assert pasted.image.getpixel((0, 0))[:3] == (255, 0, 0)
    assert (pasted.x, pasted.y) == (1, 2)
    assert pasted.opacity == 0.5
    assert pasted.blend_mode == "multiply"
    assert pasted.id != photo.id
    names = [layer.name for layer in session.document.layers]
    assert names == ["Background", "Back", "Photo", "Back copy", "Photo copy"]
    assert {layer.name for layer in session.selected_layers()} == {"Back copy", "Photo copy"}
    # A second paste clones the snapshot again, not the layer just inserted.
    pasted.image.putpixel((0, 0), (0, 0, 255, 255))
    assert session.paste_copied_layer()
    second = session.active_layer()
    assert second is not pasted
    assert second.image.getpixel((0, 0))[:3] == (255, 0, 0)
    session.undo()
    assert second not in session.document.layers
    assert pasted in session.document.layers
    assert [layer.name for layer in session.document.layers].count("Back copy") == 1

    # A picture copy sits above that picture's masks, same as Duplicate.
    mask = redaction_layer("Blur", Image.new("L", (8, 8), 255), "blur", 4, target_id=photo.id)
    session.layer_index = session.document.index_of(photo.id)
    session.add_layer(mask)
    session.layer_index = session.document.index_of(photo.id)
    session.copy_active_layer()
    session.paste_copied_layer()
    assert [layer.name for layer in session.document.layers] == [
        "Background",
        "Back",
        "Photo",
        "Blur",
        "Photo copy",
        "Back copy",
        "Photo copy",
    ]
    blur = session.document.layers[3]
    assert blur.target_id == photo.id

    # The paste above is the active layer. Copy it, then paste on another page.
    copied_name = session.active_layer().name
    session.copy_active_layer()
    session.add_group()
    assert session.paste_copied_layer()
    assert [layer.name for layer in session.document.layers] == ["Background", f"{copied_name} copy"]
    assert session.active_layer().image.getpixel((0, 0))[:3] == (0, 255, 0)


def test_copy_and_paste_keeps_text_editable():
    from pixedit.factory import text_layer
    from pixedit.model import TextInfo

    session = Session(new_document(16, 16, "white"))
    info = TextInfo("Hello", None, "Default", 18, (255, 0, 0, 255))
    session.add_layer(text_layer("Caption", info, 3, 4))
    assert session.copy_active_layer()
    session.active_layer().text.text = "Changed"
    assert session.paste_copied_layer()
    pasted = session.active_layer()
    assert pasted.kind == "text"
    assert pasted.text is not None
    assert pasted.text.text == "Hello"
    assert (pasted.x, pasted.y) == (3, 4)
    assert pasted.id != session.document.layers[1].id


def test_cut_and_delete_remove_every_selected_layer():
    """Several selected layers are cut or deleted together. A picture takes its masks."""
    from pixedit.factory import redaction_layer

    session = Session(new_document(8, 8, "white"))
    photo = raster_layer("Photo", Image.new("RGBA", (4, 4), (255, 0, 0, 255)), 1, 2)
    other = raster_layer("Other", Image.new("RGBA", (4, 4), (0, 255, 0, 255)), 3, 3)
    session.add_layer(photo)
    mask = redaction_layer("Blur", Image.new("L", (8, 8), 255), "blur", 4, target_id=photo.id)
    session.add_layer(mask)
    session.add_layer(other)
    session.select_layers([photo.id, other.id], active_id=other.id)
    steps = len(session.history.undo_stack)
    assert session.cut_active_layer()
    assert len(session.history.undo_stack) == steps + 1
    assert [layer.name for layer in session.document.layers] == ["Background"]
    assert session.paste_copied_layer()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Blur", "Other"]
    photo_copy = next(layer for layer in session.document.layers if layer.name == "Photo")
    blur = next(layer for layer in session.document.layers if layer.name == "Blur")
    assert blur.target_id == photo_copy.id
    assert {layer.name for layer in session.selected_layers()} == {"Photo", "Blur", "Other"}
    assert session.active_layer().name == "Other"

    session.delete_selected()
    assert [layer.name for layer in session.document.layers] == ["Background"]
    assert session.history.undo_stack[-1].name == "Delete layers"
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Blur", "Other"]


def test_cut_and_paste_moves_the_layer_and_its_masks():
    """Ctrl+X removes the picture and its masks. Ctrl+V puts them back with the same names."""
    from pixedit.factory import redaction_layer

    session = Session(new_document(8, 8, "white"))
    photo = raster_layer("Photo", Image.new("RGBA", (4, 4), (255, 0, 0, 255)), 1, 2)
    session.add_layer(photo)
    mask = redaction_layer("Blur", Image.new("L", (8, 8), 255), "blur", 4, target_id=photo.id)
    session.add_layer(mask)
    session.layer_index = session.document.index_of(photo.id)
    steps = len(session.history.undo_stack)
    assert session.cut_active_layer()
    assert len(session.history.undo_stack) == steps + 1
    assert [layer.name for layer in session.document.layers] == ["Background"]
    assert session.paste_copied_layer()
    names = [layer.name for layer in session.document.layers]
    assert names == ["Background", "Photo", "Blur"]
    pasted = session.document.layers[1]
    blur = session.document.layers[2]
    assert pasted.id != photo.id
    assert blur.target_id == pasted.id
    assert session.active_layer() is pasted
    # A second paste clones the snapshot again. Undoing both pastes, then the
    # cut, puts the original picture and mask back.
    assert session.paste_copied_layer()
    assert [layer.name for layer in session.document.layers].count("Photo") == 2
    session.undo()
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Background"]
    session.undo()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Blur"]


def test_pasted_mask_stays_on_its_picture():
    from pixedit.factory import redaction_layer

    session = Session(new_document(8, 8, "white"))
    messages: list[str] = []
    session.set_message_handler(messages.append)
    photo = raster_layer("Photo", Image.new("RGBA", (8, 8), (1, 2, 3, 255)))
    session.add_layer(photo)
    mask = redaction_layer("Blur", Image.new("L", (8, 8), 255), "blur", 4, target_id=photo.id)
    session.add_layer(mask)
    session.layer_index = session.document.index_of(mask.id)
    assert session.copy_active_layer()
    assert session.paste_copied_layer()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Blur", "Blur copy"]
    assert session.active_layer().kind == "redaction"
    assert session.active_layer().target_id == photo.id

    session.undo()
    session.layer_index = session.document.index_of(mask.id)
    session.delete_active()
    assert session.paste_copied_layer()
    assert [layer.name for layer in session.document.layers] == ["Background", "Photo", "Blur copy"]
    assert session.active_layer().target_id == photo.id

    session.add_group()
    messages.clear()
    before = [layer.name for layer in session.document.layers]
    assert session.paste_copied_layer() is False
    assert [layer.name for layer in session.document.layers] == before
    assert any("not on this page" in message for message in messages)
