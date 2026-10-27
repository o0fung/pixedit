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

The version sits on the right of the status bar, and Help → About PixEdit (the PixEdit menu on macOS) opens a short guide to file types, Save versus Export, the output folder, plugins, and startup logs.

## What you can do

**Open.** PNG, JPEG, WebP, TIFF, BMP, and GIF. A still image is one page. Multi-frame GIF and TIFF become one page per frame, and one group tags every frame. Multi-page PDF is rendered with pypdfium2 (File → Open, then pick 72, 144, or 216 DPI). Each PDF page is its own page, tagged with one group.

**File toolbar.** A strip under the menu bar has Save, Open, Place, Export, Undo, Redo, Show folder, and About. Save, Open, Place, and Export run the same actions as the File menu. Undo and Redo run the same actions as the Edit menu. About opens the same window as Help → About PixEdit. The buttons stay labeled Undo and Redo; the menu names the step, such as Undo Brush, and the button is grayed out when that stack is empty. Save (`Ctrl+S`) writes the `.pix` project, asking for a path the first time. Open (`Ctrl+O`) replaces the current document. New (`Ctrl+N`) opens a blank canvas that starts at 1280 × 800, with templates for HD, 4K, a square, and paper pages (A4, A3, A5, and US Letter) in portrait or landscape at 72, 150, or 300 dpi, storing that resolution so the status bar matches the sheet. Place (`Ctrl+Shift+O`) adds an image as new layers and keeps the document. Export (`Ctrl+E`) writes a flattened PNG, JPEG, WebP, TIFF, BMP, GIF, or PDF. Those dialogs start in the output folder. That folder is Documents/output/PixEdit until you pick another one with File → Choose output folder…. Show folder opens it in the file manager. Choosing a new folder does not move files already saved somewhere else, and Save of a project that already has a path still writes that file.

**Text.** The Text tool (`T`) asks for the string. Font family, style, alignment, size, line spacing, and rotation are on the tool panel, and the color is the palette. Style lists the real cuts of that family, such as Regular, Bold, or Italic. Align is Left, Center, Right, or Justify. On more than one line, center and right move the shorter lines, and justify spreads the gaps on every line except the last. One line does not move, because the text is only as wide as the glyphs. Click an existing text layer to edit it. Scaling and rotating a text layer keep it editable. Flipping rasterizes it.

**Brush and eraser.** Size, hardness, and opacity. The options panel shows one stamp: the brush is the foreground color on a checkerboard, and the eraser is a hole in a solid chip. A small brush stays small; a brush larger than the swatch fills it, and the slider still shows the real size. A fast drag is filled in along the path, so the stroke stays continuous. The pointer is a circle the size of the brush (the eraser uses the same circle). `[` and `]` change the size and the circle and the swatch update. The eraser clears pixels on the active raster layer (color and alpha), or lifts a mask when a mask layer is active. A selection clips it, the same way it clips the brush. A stroke on a new or smaller layer grows a tight RGBA bitmap around the opaque pixels and places it with the layer's `(x, y)`. It is not expanded to a full-canvas transparent sheet. Brushing past that layer's edge grows the bitmap. The background, an imported full-canvas image, and a flattened image keep their size.

**Color.** The foreground swatch sits under the tools. Its Opacity slider sets the alpha used by the brush, by shape fill and stroke, and by the fill tool. The brush tool's own Opacity slider multiplies with that. The Image & Eyedrop tool (`I`) still picks a color from the flattened image when you click. Its panel adjusts the active layer: hue, saturation, brightness, contrast, highlight, and shade. The sliders preview together. Apply keeps that as one undo step, and Reset, Escape, switching tools, or selecting another layer puts the layer back. A selection limits the grade. Text is rasterized when you apply it.

**Fill.** Fill (`G`) flood-fills 4-connected pixels on the active layer, using the whole canvas. Pixels the layer bitmap does not cover count as transparent, so a click on an empty layer or outside a small drawing still fills. Each RGBA channel of a filled pixel is within the tolerance of the pixel you click. The fill stops at a pixel outside that tolerance or at the canvas edge. It also covers the soft inner edge of a brush stroke, so a closed outline does not keep a rim of the old background. The inside of a closed outline still fills, because the outline is a different pixel. Text is rasterized first, the same way the brush does it. A selection limits the fill to that intersection. Clicking outside the canvas or the selection reports that in the status bar.

**Shapes.** Shape (`U`) drags a rectangle, ellipse, or line. While the mouse is down, the canvas shows the shape that will be committed, including fill, stroke width, color, and opacity.

**Mask.** Draw a rectangle, ellipse, or freehand region (`R`) on the active picture or text layer. The effect is pixelate, blur, solid color, noise, desaturate, invert, dim, or sharpen, stored on a mask layer that belongs to that layer. Solid uses the foreground color and makes the covered area of that layer opaque. The amount slider's label and range follow the effect, and Solid hides it. The target layer's pixels stay as they are. The effect runs while compositing, and only on that layer, so pictures below it stay sharp. Hide the mask, change its opacity, or delete it and the target comes back. Layer → Apply mask bakes the effect into the target and removes the mask. Merge down, Export, and Flatten bake it too.

**Pages and groups.** A page is one canvas: its own size, layers, selection, and guides. Page order is the PDF and GIF export order. A group is a tag, not a folder. Its layers are drawn on every page tagged with that group, after the page's own layers, and clipped to that page. One page can carry several groups, so a header can come from one group and a footer from another. The same group can tag pages that are not next to each other. The last tag on a page paints on top. Layer → New group tags the page on screen. Layer → Tag pages adds or removes the selected group. A new page keeps the tags of the page it was added from.

**Layers.** New, rename (double-click the name), reorder, show/hide, opacity, and blend modes: Normal, Multiply, Screen, Overlay. Duplicate, delete, merge down, and flatten. Merge down on a picture bakes that picture's masks into it, then merges the result into the layer below, as one undo step. Merge down on a mask bakes the mask into its picture and leaves the picture in the stack. Deleting a picture also deletes its masks. A mask stays directly above its picture when you reorder. New layers start empty and content-sized. Masks stay document-sized, which keeps the effect aligned with the canvas. A layer added while a group is selected is shared by every tagged page. Flatten bakes the page, including its group layers, and clears that page's tags so the shared layers are not drawn twice. File → Place… or Layer → Place image… (`Ctrl+Shift+O`) adds an image as new layers on the current page, or a PDF as new pages. The open document is not replaced. Undo removes what was placed.

**Selection.** After you draw a selection, the status bar lists what you can do next. Edit, Select, and Layer all have Cut selection to new layer (`Ctrl+X`) and Copy selection to new layer (`Ctrl+C`). Delete or Edit → Clear selection contents clears the pixels. Select → Deselect, Edit → Deselect, `Ctrl+D`, and Esc (when you are not mid-drag) clear the selection. The toolbar groups Transform, Hand, Zoom, and Select together, then the paint tools, then Image & Eyedrop, Sizes & Crop, and Mask, then Marker and other plugins.

**Transforms.** Transform (`V`) drags the active layer. Corner handles scale it (Shift locks aspect) and the handle above the box rotates it. The tool panel flips the active layer, turns it 90°, or opens Scale and Rotate. Scale takes a percent or a pixel size, and Keep proportions links the two axes. Layer menu: scale, rotate, flip horizontal, flip vertical. Image menu: canvas size, image size, rotate or flip the whole canvas, crop to the selection, and trim transparent edges. The Sizes & Crop tool (`C`) crops when you release. A drag past the edge expands the canvas with an empty margin. The panel sets a proportion (Free, Original, 1:1, 4:3, 16:9, or Custom), and Shift locks the canvas proportion while Free is selected. Canvas size grows or trims the frame without resampling. A 3×3 grid picks the corner, side, or center that stays put, and the new margin can be empty, white, black, or the foreground color. Image size resamples the pixels with nearest, bilinear, bicubic, or Lanczos interpolation. With resample off, the pixel grid stays put and only the resolution changes. Percent, pixels, and inches stay linked while Keep proportions is on. The status bar shows the canvas size and the resolution in dots per inch.

**Selection, paint, and adjustments.** Rectangular, elliptical, and freehand selections, plus a magic wand (contiguous pixels within a per-channel tolerance). Shift adds, Alt subtracts. The selection is a tint plus a marching outline. After you draw one, the status bar notes that brush, fill, shapes, and filters stay inside it. Delete or Backspace clears those pixels on the active raster layer (text is rasterized). That does not delete the layer — the layer panel's Delete button does. Ctrl+D deselects. Image → Crop to selection crops the canvas. Edit → Clear selection contents is the same clear. The Image & Eyedrop tool (`I`) and rectangle / ellipse / line shapes (`U`). Brightness/contrast, levels, hue/saturation, desaturate, invert, blur, sharpen, and pixelate stay in the Image and Filter menus. No selection means the whole layer.

**View.** Scroll to zoom, Space or the Hand tool (`H`) to pan, rulers, Fit (`Ctrl+0`), and actual pixels (`Ctrl+1`).

**Pointer.** On the canvas, the pointer matches the tool. Fill is a bucket, and Image & Eyedrop is an eyedropper. Shape follows rectangle, ellipse, or line, and the click is the corner the drag starts from. Select and Mask follow the same region, with a dashed rectangle or ellipse, a lasso, or a magic wand. Zoom is a magnifying glass and shows a minus while Alt is held. Sizes & Crop is a crop mark. Brush and eraser stay a circle the size of the brush. Text is an I-beam, Hand is a hand, and Transform is the move arrows until the pointer is on a corner or the rotate handle. A corner shows the diagonal scale arrow for that corner, and the handle above the box shows a rotate mark. The pointer stays on that action until the drag ends. Holding Space still pans with a hand.

**History.** Undo (`Ctrl+Z`) and redo (`Ctrl+Shift+Z` or `Ctrl+Y`). The same commands are Undo and Redo on the top bar.

## Save and export

| Action | Result |
| --- | --- |
| Save (`Ctrl+S`) | A `.pix` project: a zip with `manifest.json`, one folder of PNGs per page, shared group layers, optional masks, and the selection. |
| Export (`Ctrl+E`) | A flattened PNG, JPEG, WebP, TIFF, BMP, GIF, or PDF. JPEG, BMP, GIF, and PDF are composited on white. |

Opening a PNG does not overwrite that file on Save. Save writes a `.pix` project. Use Export for a flattened image.

A mask in a project stays a mask and remembers which layer it belongs to. That layer's pixels in the file are the originals.

## Keyboard

| Key | Action |
| --- | --- |
| `V` `B` `E` `T` `R` `S` `C` `G` `I` `U` `H` `Z` | Transform, brush, eraser, text, mask, select, crop, fill, color, shape, hand, zoom |
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

The document (canvas size, ordered layers, selection) lives in `pixedit/model.py` and is independent of Qt. `pixedit/session.py` applies undoable commands. `pixedit/composite.py` flattens the stack, including mask effects. Tools in `pixedit/tools/` and the window in `pixedit/ui/` call the session. They do not own the pixels.

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
