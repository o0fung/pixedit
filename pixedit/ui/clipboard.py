"""Move RGBA images between PixEdit and the system clipboard.

The session stays free of Qt. Copy writes PNG bytes so alpha survives, and
also a platform image so another app can paste. Paste prefers that PNG, then
falls back to whatever image the platform put on the clipboard.
"""

from __future__ import annotations

import io

from PIL import Image
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QMimeData
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication


def copy_image(image: Image.Image) -> None:
    """Put ``image`` on the clipboard as PNG and as a platform image."""
    rgba = image.convert("RGBA")
    payload = io.BytesIO()
    rgba.save(payload, format="PNG")
    png = payload.getvalue()
    qimage = QImage()
    if not qimage.loadFromData(png, "PNG"):
        return
    mime = QMimeData()
    mime.setData("image/png", QByteArray(png))
    mime.setImageData(qimage)
    QApplication.clipboard().setMimeData(mime)


def image_from_clipboard() -> Image.Image | None:
    """The clipboard image, or None when there is no image.

    Flow:
    1. Read ``image/png`` when it is present. That is what Copy writes, and
       it keeps partial alpha from a feathered selection.
    2. Otherwise take the platform image (a screenshot, or a paste from
       another app) and encode it as PNG for Pillow.
    """
    clipboard = QApplication.clipboard()
    mime = clipboard.mimeData()
    if mime is not None and mime.hasFormat("image/png"):
        image = _png_bytes(bytes(mime.data("image/png")))
        if image is not None:
            return image
    qimage = clipboard.image()
    if qimage.isNull():
        return None
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not qimage.save(buffer, "PNG"):
        return None
    return _png_bytes(bytes(buffer.data()))


def _png_bytes(data: bytes) -> Image.Image | None:
    if not data:
        return None
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (OSError, ValueError):
        return None
    return image.convert("RGBA")
