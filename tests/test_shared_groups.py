"""Shared groups: one layer stack drawn on every page that carries the tag."""

from pathlib import Path

from PIL import Image

from pixedit.composite import composite
from pixedit.factory import raster_layer
from pixedit.io import load_project, save_project
from pixedit.model import Document, LayerGroup, SharedGroup, new_document, new_id
from pixedit.session import Session


def _page(name: str, color: tuple[int, int, int, int], width: int, height: int, tags: list[str]) -> LayerGroup:
    image = Image.new("RGBA", (width, height), color)
    return LayerGroup(
        id=new_id(),
        name=name,
        width=width,
        height=height,
        layers=[raster_layer(name, image)],
        shared_ids=list(tags),
    )


def _document(*pages: LayerGroup, groups: list[SharedGroup]) -> Document:
    return Document(
        pages[0].width,
        pages[0].height,
        groups=list(pages),
        active_group_id=pages[0].id,
        shared_groups=groups,
    )


def test_a_group_layer_is_drawn_on_every_tagged_page_and_clipped():
    header = raster_layer("Header", Image.new("RGBA", (4, 2), (255, 0, 0, 255)), 0, 0)
    group = SharedGroup(id="header", name="Header", layers=[header])
    wide = _page("Wide", (255, 255, 255, 255), 6, 4, ["header"])
    short = _page("Short", (0, 0, 255, 255), 3, 2, ["header"])
    plain = _page("Plain", (0, 255, 0, 255), 6, 4, [])
    document = _document(wide, short, plain, groups=[group])

    document.active_group_id = wide.id
    top = composite(document)
    assert top.size == (6, 4)
    assert top.getpixel((0, 0))[:3] == (255, 0, 0)
    assert top.getpixel((3, 1))[:3] == (255, 0, 0)
    # The header is only two pixels tall, so the page shows below it.
    assert top.getpixel((0, 2))[:3] == (255, 255, 255)

    document.active_group_id = short.id
    small = composite(document)
    assert small.size == (3, 2)
    assert small.getpixel((0, 0))[:3] == (255, 0, 0)
    assert small.getpixel((2, 1))[:3] == (255, 0, 0)

    document.active_group_id = plain.id
    assert composite(document).getpixel((0, 0))[:3] == (0, 255, 0)

    header.image.putpixel((1, 0), (0, 255, 0, 255))
    document.active_group_id = wide.id
    assert composite(document).getpixel((1, 0))[:3] == (0, 255, 0)
    document.active_group_id = short.id
    assert composite(document).getpixel((1, 0))[:3] == (0, 255, 0)


def test_several_tags_paint_in_order_and_a_group_can_skip_pages():
    header = raster_layer("Header", Image.new("RGBA", (2, 2), (255, 0, 0, 255)), 0, 0)
    footer = raster_layer("Footer", Image.new("RGBA", (2, 2), (0, 0, 255, 255)), 0, 0)
    groups = [
        SharedGroup(id="header", name="Header", layers=[header]),
        SharedGroup(id="footer", name="Footer", layers=[footer]),
    ]
    first = _page("One", (255, 255, 255, 255), 4, 4, ["header", "footer"])
    middle = _page("Two", (255, 255, 255, 255), 4, 4, ["footer"])
    last = _page("Three", (255, 255, 255, 255), 4, 4, ["header"])
    document = _document(first, middle, last, groups=groups)

    document.active_group_id = first.id
    # The footer tag is listed after the header, so it covers the same pixels.
    assert composite(document).getpixel((0, 0))[:3] == (0, 0, 255)
    document.active_group_id = middle.id
    assert composite(document).getpixel((0, 0))[:3] == (0, 0, 255)
    document.active_group_id = last.id
    assert composite(document).getpixel((0, 0))[:3] == (255, 0, 0)


def test_tag_toggle_flatten_and_round_trip(tmp_path: Path):
    session = Session(new_document(4, 4, "white"))
    session.add_shared_group()
    group = session.document.shared_groups[0]
    assert session.document.group.shared_ids == [group.id]
    session.add_blank_layer()
    shared = session.active_layer()
    assert shared is not None and shared in group.layers
    shared.image = Image.new("RGBA", (4, 1), (255, 0, 0, 255))
    shared.x = 0
    shared.y = 0
    assert composite(session.document).getpixel((0, 0))[:3] == (255, 0, 0)

    session.add_group()
    assert session.document.group.shared_ids == []
    assert composite(session.document).getpixel((0, 0))[:3] == (255, 255, 255)

    second = session.document.active_group_id
    first = session.document.groups[0].id
    session.toggle_page_tags(group.id, [second])
    assert session.document.group.shared_ids == [group.id]
    assert composite(session.document).getpixel((0, 0))[:3] == (255, 0, 0)
    session.toggle_page_tags(group.id, [second])
    assert group.id not in session.document.group.shared_ids
    assert composite(session.document).getpixel((0, 0))[:3] == (255, 255, 255)
    session.activate_group(first)
    assert composite(session.document).getpixel((0, 0))[:3] == (255, 0, 0)
    session.undo()
    session.activate_group(second)
    assert group.id in session.document.group.shared_ids

    session.flatten()
    assert session.document.group.shared_ids == []
    assert composite(session.document).getpixel((0, 0))[:3] == (255, 0, 0)
    session.activate_group(first)
    assert group.id in session.document.group.shared_ids
    assert composite(session.document).getpixel((0, 0))[:3] == (255, 0, 0)
    session.undo()
    session.activate_group(second)
    assert group.id in session.document.group.shared_ids

    path = tmp_path / "tagged.pix"
    save_project(session.document, str(path))
    loaded = load_project(str(path))
    assert len(loaded.shared_groups) == 1
    assert loaded.groups[0].shared_ids == [loaded.shared_groups[0].id]
    assert loaded.groups[1].shared_ids == [loaded.shared_groups[0].id]
    loaded.active_group_id = loaded.groups[1].id
    assert composite(loaded).getpixel((0, 0))[:3] == (255, 0, 0)

    session.delete_shared_groups([group.id])
    assert session.document.shared_groups == []
    assert all(page.shared_ids == [] for page in session.document.groups)
    session.undo()
    assert session.document.shared_groups[0].id == group.id
    assert group.id in session.document.groups[0].shared_ids


def test_moving_a_page_layer_into_a_group_shares_it_and_keeps_other_tag_order():
    session = Session(new_document(4, 4, "white"))
    session.add_shared_group()
    header = session.document.shared_groups[0]
    footer_layer = raster_layer("Mark", Image.new("RGBA", (4, 1), (0, 0, 255, 255)), 0, 3)
    session._shared_id = None
    session.add_layer(footer_layer, name="Add layer")
    assert footer_layer in session.document.layers
    session.add_shared_group()
    footer = session.document.shared_groups[1]
    # The new group was appended, so it paints above the header.
    assert session.document.group.shared_ids == [header.id, footer.id]
    moved = session.transfer_layers([footer_layer.id], "page", session.document.group.id, "shared", footer.id, 0)
    assert moved is True
    assert footer_layer in footer.layers
    assert footer_layer not in session.document.group.layers
    assert composite(session.document).getpixel((0, 3))[:3] == (0, 0, 255)
    session.add_group()
    second = session.document.active_group_id
    assert session.document.group.shared_ids == []
    # The new page starts bare. Give it the first page's tags, in paint order,
    # so the toggles below can check that an existing tag stays where it was.
    session.toggle_page_tags(header.id, [second])
    session.toggle_page_tags(footer.id, [second])
    assert session.document.group.shared_ids == [header.id, footer.id]
    session.toggle_page_tags(header.id, [second])
    assert session.document.groups[0].shared_ids == [header.id, footer.id]
    assert session.document.group.shared_ids == [footer.id]
    # A mixed selection adds the missing tag and leaves a page that already has it alone.
    session.toggle_page_tags(header.id, [session.document.groups[0].id, second])
    assert session.document.groups[0].shared_ids == [header.id, footer.id]
    assert session.document.group.shared_ids == [footer.id, header.id]


def test_group_and_tag_buttons_add_and_remove_a_tag():
    from PyQt6.QtWidgets import QApplication, QToolButton

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(4, 4, "white"))
    panel = LayerPanel(session)
    panel.show()
    app.processEvents()
    icons = {button.accessibleName(): button for button in panel.findChildren(QToolButton)}
    assert panel._groups_empty.isVisibleTo(panel)
    icons["New tag"].click()
    app.processEvents()
    assert not panel._groups_empty.isVisibleTo(panel)
    assert panel.groups.topLevelItemCount() == 1
    assert len(session.document.shared_groups) == 1
    group_id = session.document.shared_groups[0].id
    assert session.document.group.shared_ids == [group_id]
    assert panel._selected_shared == [group_id]

    icons["Page"].click()
    app.processEvents()
    assert len(session.document.groups) == 2
    assert session.document.group.shared_ids == []
    icons["Tag pages"].click()
    app.processEvents()
    assert session.document.group.shared_ids == [group_id]
    assert session.document.groups[0].shared_ids == [group_id]

    photo = session.document.groups[0].layers[0]
    page = panel.list.topLevelItem(0)
    assert page is not None and page.childCount() >= 1
    layer_row = page.child(0)
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(layer_row)
    layer_row.setSelected(True)
    panel.list.blockSignals(False)
    header = panel.groups.topLevelItem(0)
    assert header is not None
    assert panel.prepare_cross_drop(panel.list, header, 0)
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(page)
    page.setSelected(True)
    panel.list.blockSignals(False)
    assert panel.prepare_cross_drop(panel.list, header, 0) is False
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(layer_row)
    layer_row.setSelected(True)
    panel.list.blockSignals(False)
    assert panel.prepare_cross_drop(panel.list, header, 0)
    panel._apply_pending_drop()
    app.processEvents()
    assert all(layer.id != photo.id for layer in session.document.groups[0].layers)
    assert any(layer.id == photo.id for layer in session.document.shared_groups[0].layers)
    panel.close()
    app.processEvents()


def test_selected_layers_move_between_a_page_and_a_tag_together():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(4, 4, "white"))
    session.document.layers[0].name = "Back"
    mid = raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255)))
    front = raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
    session.add_layer(mid)
    session.add_layer(front)
    session.add_shared_group()
    tag_id = session.document.shared_groups[0].id
    session._shared_id = None
    panel = LayerPanel(session)
    panel.show()
    app.processEvents()
    panel.refresh()
    page = panel.list.topLevelItem(0)
    tag = panel.groups.topLevelItem(0)
    assert page is not None and tag is not None
    front_row = page.child(0)
    mid_row = page.child(1)
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(front_row)
    front_row.setSelected(True)
    mid_row.setSelected(True)
    panel.list.blockSignals(False)
    panel._on_drag_started()
    assert panel.prepare_cross_drop(panel.list, tag, 0)
    panel._apply_pending_drop()
    app.processEvents()
    tag_layers = session.document.shared_groups[0].layers
    assert [layer.name for layer in tag_layers] == ["Mid", "Front"]
    assert [layer.name for layer in session.document.layers] == ["Back"]
    assert session._shared_id == tag_id
    assert [layer.name for layer in session.selected_layers()] == ["Mid", "Front"]
    assert session.active_layer().name == "Front"

    panel._on_drag_started()
    page = panel.list.topLevelItem(0)
    assert page is not None
    assert panel.prepare_cross_drop(panel.groups, page, 0)
    panel._apply_pending_drop()
    app.processEvents()
    assert session.document.shared_groups[0].layers == []
    assert [layer.name for layer in session.document.layers] == ["Back", "Mid", "Front"]
    assert session._shared_id is None
    assert [layer.name for layer in session.selected_layers()] == ["Mid", "Front"]
    assert session.active_layer().name == "Front"
    panel.close()
    app.processEvents()


def test_layers_inside_a_tag_drag_as_one_block():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(4, 4, "white"))
    session.add_shared_group()
    session.add_layer(raster_layer("Base", Image.new("RGBA", (4, 4), (0, 0, 255, 255))))
    session.add_layer(raster_layer("Low", Image.new("RGBA", (4, 4), (255, 0, 0, 255))))
    session.add_layer(raster_layer("High", Image.new("RGBA", (4, 4), (0, 255, 0, 255))))
    panel = LayerPanel(session)
    panel.refresh()
    tag = panel.groups.topLevelItem(0)
    assert tag is not None
    high = tag.child(0)
    low = tag.child(1)
    panel.groups.blockSignals(True)
    panel.groups.clearSelection()
    panel.groups.setCurrentItem(high)
    high.setSelected(True)
    low.setSelected(True)
    panel.groups.blockSignals(False)
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.blockSignals(False)
    panel._on_drag_started()
    taken = tag.takeChild(0)
    assert taken is not None
    tag.addChild(taken)
    panel._apply_pending_drop()
    assert [layer.name for layer in session.document.shared_groups[0].layers] == ["Low", "High", "Base"]
    assert [layer.name for layer in session.selected_layers()] == ["Low", "High"]
    assert session.active_layer().name == "High"
    panel.close()


def test_tag_pages_uses_every_highlighted_page_header():
    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.layers_panel import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(4, 4, "white"))
    session.rename_group(session.document.groups[0].id, "One")
    session.add_shared_group()
    tag_id = session.document.shared_groups[0].id
    session.toggle_page_tags(tag_id, [session.document.groups[0].id])
    assert session.document.groups[0].shared_ids == []
    session.add_group()
    session.rename_group(session.document.active_group_id, "Two")
    panel = LayerPanel(session)
    panel.show()
    app.processEvents()
    panel.refresh()
    tag = panel.groups.topLevelItem(0)
    assert tag is not None
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.blockSignals(False)
    panel.groups.blockSignals(True)
    panel.groups.clearSelection()
    panel.groups.setCurrentItem(tag)
    tag.setSelected(True)
    panel.groups.blockSignals(False)
    panel._on_selection()
    assert panel._tag_group_id == tag_id

    two = panel.list.topLevelItem(1)
    assert two is not None and two.child(0) is not None
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    panel.list.setCurrentItem(two.child(0))
    two.child(0).setSelected(True)
    panel.list.blockSignals(False)
    panel._on_selection()
    assert panel._selected_groups == []
    panel.tag_selection()
    assert session.document.groups[0].shared_ids == []
    assert session.document.groups[1].shared_ids == [tag_id]
    session.undo()
    assert session.document.groups[1].shared_ids == []

    one = panel.list.topLevelItem(0)
    assert one is not None
    panel.list.blockSignals(True)
    panel.list.clearSelection()
    # setCurrentItem clears the other rows, so the headers are selected after it.
    assert two.child(0) is not None
    panel.list.setCurrentItem(two.child(0))
    one.setSelected(True)
    two.setSelected(True)
    two.child(0).setSelected(True)
    panel.list.blockSignals(False)
    panel._on_selection()
    assert panel._selected_groups == []
    panel.tag_selection()
    assert session.document.groups[0].shared_ids == [tag_id]
    assert session.document.groups[1].shared_ids == [tag_id]
    panel.refresh()
    assert panel.list.topLevelItem(0).isSelected()
    assert panel.list.topLevelItem(1).isSelected()
    panel.tag_selection()
    assert session.document.groups[0].shared_ids == []
    assert session.document.groups[1].shared_ids == []
    session.undo()
    assert session.document.groups[0].shared_ids == [tag_id]
    assert session.document.groups[1].shared_ids == [tag_id]
    panel.close()
    app.processEvents()
