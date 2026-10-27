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
    assert session.document.group.shared_ids == [group.id]
    assert composite(session.document).getpixel((0, 0))[:3] == (255, 0, 0)

    second = session.document.active_group_id
    first = session.document.groups[0].id
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
    icons["Group"].click()
    app.processEvents()
    assert len(session.document.shared_groups) == 1
    group_id = session.document.shared_groups[0].id
    assert session.document.group.shared_ids == [group_id]
    assert panel._selected_shared == [group_id]

    icons["Page"].click()
    app.processEvents()
    assert len(session.document.groups) == 2
    assert session.document.group.shared_ids == [group_id]
    icons["Tag"].click()
    app.processEvents()
    assert session.document.group.shared_ids == []
    assert session.document.groups[0].shared_ids == [group_id]
    panel.close()
    app.processEvents()
