By Yeung LF developed with CursorAi

PixEdit is a layered desktop image editor. Launch it with `pix`, or with `python -m pixedit`. Passing a path, as in `pix photo.png`, opens that file.

## Open

PNG, JPEG, WebP, TIFF, BMP, GIF, and PDF. A still image is one page. Multi-frame GIF and TIFF become one page per frame, tagged with one group. A multi-page PDF is rendered at 72, 144, or 216 DPI, one page per PDF page, tagged with one group. A group's layers, such as a header or footer, are drawn on every page with that tag. A `.pix` project opens with its pages, groups, masks, and selection.

## Save and export

Save writes a `.pix` project: a zip of the pages, shared groups, masks, and the selection. Opening a PNG does not overwrite that file on Save. Use Export for a flattened PNG, JPEG, WebP, TIFF, BMP, GIF, or PDF. JPEG, BMP, GIF, and PDF are composited on white. PDF and animated GIF include each page's group layers.

## Output folder

Open, Place, Save, and Export start in Documents/output/PixEdit until you pick another folder with File → Choose output folder. Show folder opens that folder in the file manager. Choosing a new folder does not move files already saved somewhere else.

## Keyboard

Help → Keyboard shortcuts lists the keys. The left toolbar shows the same letter on each tool button, for example Brush (B).

## Plugins

A user plugin is a single `.py` file in `~/.pixedit/plugins/`. Restart, or choose Plugins → Reload plugins. The shipped example adds Filter → Vignette and the Marker tool.

## If PixEdit fails to start

Startup logs are written to `~/Library/Logs/pixedit` on macOS, and to `~/.pixedit/logs` on Linux and Windows.
