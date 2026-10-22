# PixEdit

PixEdit is a layered desktop image editor. Install the package, then launch it with `pix` — the same pattern as a short console script opening a PyQt6 window.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pix
```

`pix photo.png` opens a file. `python -m pixedit` is the same entry point.

## What you can do

**Open.** PNG, JPEG, WebP, TIFF, BMP, and GIF. Multi-frame GIF and TIFF become one layer per frame; only the first frame is visible. Multi-page PDF is rendered with pypdfium2 (File → Open, then pick 72, 144, or 216 DPI). Each page is a layer.

**File toolbar.** A strip under the menu bar has Save, Open, Place, and Export. They run the same actions as the File menu. Save (`Ctrl+S`) writes the `.pix` project, asking for a path the first time. Open (`Ctrl+O`) replaces the current document. Place (`Ctrl+Shift+O`) adds an image as new layers and keeps the document. Export (`Ctrl+E`) writes a flattened PNG, JPEG, or WebP.

**Text.** The Text tool (`T`) asks for the string, font, size, color, and rotation. Click an existing text layer to edit it. Scaling and rotating a text layer keep it editable. Flipping rasterizes it.

**Brush and eraser.** Size, hardness, and opacity. A fast drag is filled in along the path, so the stroke stays continuous. The pointer is a circle the size of the brush (the eraser uses the same circle). `[` and `]` change the size and the circle updates. The eraser clears pixels on the active raster layer (color and alpha), or lifts a redaction mask when a redaction layer is active. A selection clips it, the same way it clips the brush. A stroke on a new or smaller layer grows a tight RGBA bitmap around the opaque pixels and places it with the layer's `(x, y)`. It is not expanded to a full-canvas transparent sheet. Brushing past that layer's edge grows the bitmap. The background, an imported full-canvas image, and a flattened image keep their size.

**Color.** The foreground swatch sits under the tools. Its Opacity slider sets the alpha used by the brush, by shape fill and stroke, and by the fill tool. The brush tool's own Opacity slider multiplies with that.

**Fill.** Fill (`G`) flood-fills 4-connected pixels on the active layer. Each RGBA channel of a filled pixel is within the tolerance of the pixel you click. The fill also covers the soft inner edge of a brush stroke, so a closed outline does not keep a rim of the old background. Text is rasterized first, the same way the brush does it. A selection limits the fill to that intersection. Clicking outside the layer or the selection reports that in the status bar.

**Shapes.** Shape (`U`) drags a rectangle, ellipse, or line. While the mouse is down, the canvas shows the shape that will be committed, including fill, stroke width, color, and opacity.

**Privacy redaction.** Draw a rectangle, ellipse, or freehand region (`R`). The effect is Gaussian blur or a pixelate/mosaic, stored as a mask layer. The pixels underneath are not rewritten. The effect is applied while compositing, so you can hide the layer, change its opacity, or delete it and the original image comes back. Export and Flatten bake the effect.

**Layers.** New, rename (double-click the name), reorder, show/hide, opacity, and blend modes: Normal, Multiply, Screen, Overlay. Duplicate, delete, merge down, and flatten. Merge down bakes the upper layer into the one beneath it, including a redaction mask, as one undo step. New layers start empty and content-sized. Redaction masks stay document-sized, which keeps blur and pixelate aligned with the canvas. File → Place… or Layer → Place image… (`Ctrl+Shift+O`) adds a PNG, JPEG, WebP, TIFF, BMP, GIF, or PDF as new layers centered on the current document. The open document is not replaced. Undo removes the placed layers.

**Selection.** After you draw a selection, the status bar lists what you can do next. Edit, Select, and Layer all have Cut selection to new layer (`Ctrl+X`) and Copy selection to new layer (`Ctrl+C`). Delete or Edit → Clear selection contents clears the pixels. Select → Deselect, Edit → Deselect, `Ctrl+D`, and Esc (when you are not mid-drag) clear the selection. The toolbar groups Move, Hand, Zoom, Select, and Eyedrop together, then the paint tools, then Crop and Redact, then Marker and other plugins.

**Transforms.** Move (`V`) drags the active layer. Corner handles scale it (Shift locks aspect) and the handle above the box rotates it. Layer menu: scale, rotate, flip horizontal, flip vertical. Image menu: rotate or flip the whole canvas, crop to the selection, and trim transparent edges. The Crop tool (`C`) crops the canvas when you release.

**Selection, paint, and adjustments.** Rectangular, elliptical, and freehand selections, plus a magic wand (contiguous pixels within a per-channel tolerance). Shift adds, Alt subtracts. The selection is a tint plus a marching outline. After you draw one, the status bar notes that brush, fill, shapes, and filters stay inside it. Delete or Backspace clears those pixels on the active raster layer (text is rasterized). That does not delete the layer — the layer panel's Delete button does. Ctrl+D deselects. Image → Crop to selection crops the canvas. Edit → Clear selection contents is the same clear. Eyedropper (`I`) and rectangle / ellipse / line shapes (`U`). Brightness/contrast, levels, hue/saturation, desaturate, invert, blur, sharpen, and pixelate. No selection means the whole layer.

**View.** Scroll to zoom, Space or the Hand tool (`H`) to pan, rulers, Fit (`Ctrl+0`), and actual pixels (`Ctrl+1`).

**History.** Undo (`Ctrl+Z`) and redo (`Ctrl+Shift+Z` or `Ctrl+Y`).

## Save and export

| Action | Result |
| --- | --- |
| Save (`Ctrl+S`) | A `.pix` project: a zip with `manifest.json`, PNG layer assets, optional masks, and the selection. |
| Export (`Ctrl+E`) | A flattened PNG, JPEG, or WebP. JPEG is composited on white. |

Opening a PNG does not overwrite that file on Save. Save writes a `.pix` project. Use Export for a flattened image.

A redaction in a project stays a mask. The source layer's pixels in the file are the originals.

## Keyboard

| Key | Action |
| --- | --- |
| `V` `B` `E` `T` `R` `M` `C` `G` `I` `U` `H` `Z` | Move, brush, eraser, text, redact, select, crop, fill, eyedropper, shape, hand, zoom |
| `[` `]` | Smaller / larger brush |
| `Ctrl+N` `Ctrl+O` | New, open |
| `Ctrl+S` `Ctrl+Shift+S` | Save project, save as |
| `Ctrl+E` | Export flattened |
| `Ctrl+Z` `Ctrl+Y` | Undo, redo |
| `Ctrl+A` `Ctrl+D` `Esc` `Ctrl+Shift+I` | Select all, deselect (`Ctrl+D` or `Esc`), invert selection |
| `Ctrl+X` `Ctrl+C` | Cut or copy the selection onto a new layer |
| `Delete` `Backspace` | Clear pixels inside the selection |
| `Ctrl+Shift+O` | Place an image as new layers |
| `Ctrl+J` `Ctrl+Shift+N` | Duplicate layer, new layer |
| `Ctrl+Shift+E` | Merge down |
| `Ctrl+0` `Ctrl+1` `Ctrl++` `Ctrl+-` | Fit, 100%, zoom in, zoom out |

Help → Keyboard shortcuts lists these inside the app. The left toolbar shows the same letter on each tool button, for example Brush (B). Marker is an example plugin and has no shortcut.

## Plugins

Plugins register commands and canvas tools without editing the editor. `register(api)` is the only hook. PixEdit loads:

1. The `pixedit.plugins` entry-point group.
2. The shipped example, if that entry point is not installed yet.
3. Every `*.py` file in `~/.pixedit/plugins/`.

The shipped example adds **Filter → Vignette** and a **Marker** tool. Marker is a stamp demo, not a second editing mode: each click draws a plus (and a small ring) on the active raster or text layer and the status bar says “Stamped marker”. There is no next step. Source: `pixedit/plugins/examples/basic.py`.

A user plugin is a single file:

```python
# ~/.pixedit/plugins/invert_extra.py
from pixedit.model import MutateLayerCommand, apply_visual, visual_state

def invert_twice(session):
    layer = session.active_layer()
    if layer is None or layer.kind == "redaction":
        session.show_message("Pick a raster or text layer.")
        return
    before = visual_state(layer)
    image = layer.image.copy()
    # ... edit `image`, then commit one undo step:
    layer.image = image
    layer.kind = "raster"
    layer.text = None
    new_state = visual_state(layer)
    apply_visual(layer, before)
    session.commit(MutateLayerCommand(layer, new_state, name="Invert twice"))

def register(api):
    api.add_command(
        command_id="mine.invert_twice",
        label="Invert twice",
        menu="Filter",
        callback=invert_twice,
    )
```

Restart, or choose Plugins → Reload plugins. A canvas tool subclasses `pixedit.tools.base.Tool` (`id`, `label`, `pointer_down` / `pointer_move` / `pointer_up`, optional `option_schema`) and is registered with `api.add_tool(tool)`.

To ship a plugin inside another package:

```toml
[project.entry-points."pixedit.plugins"]
my_plugin = "my_package.module:register"
```

Plugins should edit the document through `session.commit(...)` so undo works. They do not need to import Qt. A plugin that fails to load is reported and skipped.

## Layout

The document (canvas size, ordered layers, selection) lives in `pixedit/model.py` and is independent of Qt. `pixedit/session.py` applies undoable commands. `pixedit/composite.py` flattens the stack, including redaction. Tools in `pixedit/tools/` and the window in `pixedit/ui/` call the session. They do not own the pixels.

## Tests and headless use

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
pix --check
```

`pytest` covers compositing, the layer stack, blur and pixelate masks, transforms, paint, selection, project IO, and a headless window construct. `pix --check` builds the window on the offscreen Qt platform and exits.

`pix` itself needs a display (`DISPLAY` or `WAYLAND_DISPLAY` on Linux). On a machine without one it prints that and exits instead of crashing. Set `QT_QPA_PLATFORM=offscreen` only for the check. Startup failures are written under `~/.pixedit/logs/` (`~/Library/Logs/pixedit` on macOS).

## Requirements

Python 3.10 or newer. PyQt6, Pillow, numpy, pypdfium2, and pytest are listed in both `requirements.txt` and `pyproject.toml`. Change a dependency in both files.
