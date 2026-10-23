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
    from PyQt6.QtCore import QModelIndex
    from PyQt6.QtWidgets import QApplication, QPushButton

    from pixedit.ui.panels import LayerPanel

    app = QApplication.instance() or QApplication([])
    session = Session(new_document(4, 4, "white"))
    session.document.layers[0].name = "Back"
    session.add_layer(raster_layer("Mid", Image.new("RGBA", (4, 4), (0, 255, 0, 255))))
    session.add_layer(raster_layer("Front", Image.new("RGBA", (4, 4), (255, 0, 0, 255))))
    panel = LayerPanel(session, lambda: None)
    panel.refresh()
    labels = [button.text() for button in panel.findChildren(QPushButton)]
    assert "Merge down" in labels
    merge = next(button for button in panel.findChildren(QPushButton) if button.text() == "Merge down")
    assert "below" in merge.toolTip()
    assert "Flatten" in merge.toolTip()
    # Visual order is Front, Mid, Back. Drop Front at the bottom of the list.
    assert panel.list.model().moveRow(QModelIndex(), 0, QModelIndex(), panel.list.count())
    panel.list.drag_finished.emit()
    app.processEvents()
    assert [layer.name for layer in session.document.layers] == ["Front", "Back", "Mid"]
    assert session.layer_index == 0
    shown = [panel.list.item(row).text().split()[0] for row in range(panel.list.count())]
    assert shown == ["Mid", "Back", "Front"]
    panel.close()
