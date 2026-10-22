"""Constructors for the layer kinds the editor creates."""

from __future__ import annotations

from PIL import Image

from pixedit.model import Layer, RedactionInfo, TextInfo, new_id
from pixedit.textutil import render_text


def raster_layer(name: str, image: Image.Image, x: int = 0, y: int = 0) -> Layer:
    return Layer(
        id=new_id(),
        name=name,
        kind="raster",
        image=image.convert("RGBA"),
        x=int(x),
        y=int(y),
    )


def blank_layer(name: str, width: int, height: int, fill: tuple[int, int, int, int] = (0, 0, 0, 0)) -> Layer:
    """A document-sized plate (flatten, or an explicit full-canvas fill)."""
    return raster_layer(name, Image.new("RGBA", (width, height), fill))


def empty_raster_layer(name: str) -> Layer:
    """A content-sized blank layer. Painting grows it around the new marks."""
    return raster_layer(name, Image.new("RGBA", (1, 1), (0, 0, 0, 0)), 0, 0)


def text_layer(name: str, info: TextInfo, x: int, y: int) -> Layer:
    return Layer(
        id=new_id(),
        name=name,
        kind="text",
        image=render_text(info),
        x=int(x),
        y=int(y),
        text=info,
    )


def redaction_layer(
    name: str,
    mask: Image.Image,
    kind: str,
    amount: float,
    x: int = 0,
    y: int = 0,
) -> Layer:
    return Layer(
        id=new_id(),
        name=name,
        kind="redaction",
        image=Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
        x=int(x),
        y=int(y),
        mask=mask.convert("L"),
        redaction=RedactionInfo(kind=kind, amount=float(amount)),
    )
