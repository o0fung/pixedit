"""Guides, blend-free frame export, and dropping a file."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pixedit.factory import raster_layer
from pixedit.frames import (
    export_animated_gif,
    frame_kind,
    letterbox_offset,
    shared_frame_zoom,
)
from pixedit.guides import SNAP_DISTANCE, rotate_guides_cw, snap_delta, snap_point
from pixedit.io import load_project, save_project
from pixedit.model import Document, LayerGroup, new_document, new_id
from pixedit.session import Session
from pixedit.tools.base import PointerEvent
from pixedit.tools.shape import ShapeTool


def test_snap_pulls_a_point_and_a_box_edge_and_can_be_off():
    assert snap_point(10, 20, (12,), (100,), True) == (12.0, 20.0)
    assert snap_point(10, 20, (12,), (100,), False) == (10.0, 20.0)
    assert snap_point(10, 20, (), (), True) == (10.0, 20.0)
    # The right edge of a 10px box moved by 11 lands 1px from a guide at 20.
    dx, dy = snap_delta(11, 0, (0, 0, 10, 10), (20,), (), True)
    assert dx == 10
    assert dy == 0
    assert SNAP_DISTANCE == 6


def test_clockwise_guides_follow_the_canvas_pixels():
    # A vertical guide at x=1 on a 4×2 page becomes horizontal at y=1.
    vertical, horizontal = rotate_guides_cw((1,), (), 2)
    assert vertical == ()
    assert horizontal == (1,)


def test_crop_shifts_guides_and_undo_puts_them_back():
    session = Session(new_document(40, 30, "white"))
    session.document.group.guides_v = (10, 30)
    session.document.group.guides_h = (5,)
    session.crop_to(4, 2, 20, 20)
    assert session.document.group.guides_v == (6, 26)
    assert session.document.group.guides_h == (3,)
    session.undo()
    assert session.document.group.guides_v == (10, 30)
    assert session.document.group.guides_h == (5,)


def test_shape_snaps_to_a_guide_until_snap_is_off():
    session = Session(new_document(40, 40, "transparent"))
    session.document.group.guides_v = (20,)
    session.foreground = (0, 255, 0, 255)
    session.options["shape_kind"] = "rect"
    session.options["shape_filled"] = True
    session.options["stroke_width"] = 1
    tool = ShapeTool()
    tool.pointer_down(session, PointerEvent(18, 8))
    tool.pointer_up(session, PointerEvent(30, 20))
    layer = session.active_layer()
    assert layer.image.getpixel((20 - layer.x, 12 - layer.y))[1] > 200
    assert layer.image.getpixel((18 - layer.x, 12 - layer.y))[3] == 0

    session.undo()
    session.options["snap"] = False
    tool.pointer_down(session, PointerEvent(18, 8))
    tool.pointer_up(session, PointerEvent(30, 20))
    layer = session.active_layer()
    assert layer.image.getpixel((18 - layer.x, 12 - layer.y))[1] > 200


def test_guides_survive_a_project_round_trip(tmp_path: Path):
    group = LayerGroup(
        id=new_id(),
        name="Page",
        width=8,
        height=6,
        layers=[raster_layer("Ink", Image.new("RGBA", (8, 6), (1, 2, 3, 255)))],
        guides_v=(2, 5),
        guides_h=(1,),
    )
    document = Document(8, 6, groups=[group], active_group_id=group.id)
    path = tmp_path / "guides.pix"
    save_project(document, str(path))
    loaded = load_project(str(path))
    assert loaded.group.guides_v == (2, 5)
    assert loaded.group.guides_h == (1,)


def test_animated_gif_pads_a_smaller_page_and_refuses_a_still(tmp_path: Path):
    wide = LayerGroup(
        id=new_id(),
        name="Wide",
        width=8,
        height=4,
        layers=[raster_layer("Wide", Image.new("RGBA", (8, 4), (255, 0, 0, 255)))],
    )
    small = LayerGroup(
        id=new_id(),
        name="Small",
        width=4,
        height=4,
        layers=[raster_layer("Small", Image.new("RGBA", (4, 4), (0, 0, 255, 255)))],
    )
    document = Document(8, 4, groups=[wide, small], active_group_id=wide.id)
    assert frame_kind(document) == "pages"
    path = tmp_path / "frames.gif"
    notes = export_animated_gif(document, str(path), 40)
    assert any("Small" in note and "4 × 4" in note for note in notes)
    with Image.open(path) as gif:
        assert gif.n_frames == 2
        assert gif.size == (8, 4)
        gif.seek(1)
        assert gif.size == (8, 4)
    zoom = shared_frame_zoom(400, 300, [(8, 4), (4, 4)], ruler=0, margin=0)
    small_offset, _y = letterbox_offset(400, 300, 4, 4, zoom, ruler=0)
    wide_offset, _wide_y = letterbox_offset(400, 300, 8, 4, zoom, ruler=0)
    assert small_offset > wide_offset

    still = new_document(4, 4, "white")
    assert frame_kind(still) is None
    try:
        export_animated_gif(still, str(tmp_path / "no.gif"), 40)
    except ValueError as exc:
        assert "Nothing to export" in str(exc)
    else:
        raise AssertionError("a still document should not export as an animation")


def test_hidden_layers_are_gif_frames_and_a_drop_places_or_opens(tmp_path: Path):
    session = Session(new_document(6, 6, "white"))
    session.add_blank_layer()
    hidden = session.document.layers[-1]
    hidden.image = Image.new("RGBA", (6, 6), (255, 0, 0, 255))
    hidden.x = 0
    hidden.y = 0
    hidden.visible = False
    assert frame_kind(session.document) == "layers"
    path = tmp_path / "frames.gif"
    notes = export_animated_gif(session.document, str(path), 30)
    assert notes == []
    with Image.open(path) as gif:
        assert gif.n_frames == 2
    # Export restored the visibility it borrowed.
    assert session.document.layers[0].visible is True
    assert session.document.layers[1].visible is False

    from PyQt6.QtWidgets import QApplication

    from pixedit.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    image_path = tmp_path / "spot.png"
    Image.new("RGBA", (3, 3), (0, 255, 0, 255)).save(image_path)
    before = len(window.session.document.layers)
    window._drop_paths([str(image_path)])
    assert len(window.session.document.layers) == before + 1
    assert window.session.active_layer().name != ""

    project = tmp_path / "page.pix"
    save_project(new_document(5, 5, "black"), str(project))
    window.session.dirty = False
    kept_id = window.session.document.groups[0].id
    kept_path = window.session.path
    group_count = len(window.session.document.groups)
    window._drop_paths([str(project)])
    app.processEvents()
    # The open document stays. The project is a new group at the end.
    assert window.session.path == kept_path
    assert window.session.document.groups[0].id == kept_id
    assert len(window.session.document.groups) == group_count + 1
    assert window.session.document.active_group_id == window.session.document.groups[-1].id
    assert window.session.document.layers[0].image.getpixel((0, 0))[:3] == (0, 0, 0)

    window.session.add_group()
    home = window.session.document.active_group_id
    window._toggle_play()
    window._play_timer.stop()
    assert window._playing
    assert window.canvas.letterbox
    shown = window.session.document.active_group_id
    widths = {group.width for group in window.session.document.groups}
    window._advance_play()
    assert window.session.document.active_group_id != shown
    assert {group.width for group in window.session.document.groups} == widths
    window._stop_play()
    assert not window._playing
    assert not window.canvas.letterbox
    assert window.session.document.active_group_id == home

    labels = []
    for action in window.menuBar().actions():
        menu = action.menu()
        if menu is None:
            continue
        labels.extend(child.text() for child in menu.actions())
    assert "Export animated GIF…" in labels
    assert "Play frames" in labels
    assert "Snap to guides" in labels
    assert "gradient" not in window.tool_by_id
    window.session.dirty = False
    window.close()
    app.processEvents()
