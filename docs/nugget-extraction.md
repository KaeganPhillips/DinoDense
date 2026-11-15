# Nugget Extraction

How a scan of a dino-nugget becomes a shape file in `data/shapes/`. The extraction tool lives in `tools/extract/` and is run on one scan at a time.

## Overview

```
scan (jpg)
  |
  |-- DimeDetector ----> where the dime is, so the scale (mm per pixel)
  |
  |-- NuggetFinder ----> the nugget outline, in pixels
  |
  |-- NuggetShape -----> simplified polygon, in mm, centered on its centroid
  |
  |-- OverlayRenderer -> picture of the dime and polygon on the scan
  |
  v
data/shapes/<name>.json        data/overlays/<name>.jpg (also shown on screen)
```

`NuggetExtractor` runs these steps in order. The shape is named after the scan file, so `data/scans/trex.jpg` becomes `data/shapes/trex.json`.

## Capturing a Scan

Each scan holds one nugget and a dime on white paper. The dime is the size reference.

- Use a document-scanner mode or shoot straight down, so there is no perspective.
- Light it softly. A soft shadow beside the nugget is fine, but a hard one can look like part of the nugget.
- Keep the whole nugget and the whole dime in the picture, with the dime clear of the nugget.

Scans may have any resolution, the scale always comes from the dime and never from the pixel size.

## Running It

Setup, on Windows, from the repository root:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r tools\extract\requirements.txt
```

Extract a shape:

```
.venv\Scripts\python tools\extract\nugget_extractor.py data\scans\trex.jpg
```

This prints the scale and size it found, saves the shape and the overlay, and opens the overlay in a window. Press any key or close the window to finish. Add `--no-show` to skip the window verification. Running it again overwrites the earlier shape and overlay, so if an overlay looks wrong, fix the scan and run it again.

If anything fails, the tool prints `Cannot extract: ...` with the reason and exits with code 1.

## How It Works

Each class is in its own file in `tools/extract/`. All pixel positions are measured in the scan's pixels, with the y axis pointing down.

### DimeDetector (`dime_detector.py`)

`DimeDetector.detect(image)` finds the dime and returns a `DimeDetection` with the circle's `center_x_px`, `center_y_px` and `radius_px`. `DimeDetection` also provides `diameter_px` and `mm_per_px`, the scale of the whole scan, which is the dime's real diameter (`DIME_DIAMETER_MM`, 17.91 mm) divided by its diameter in pixels. If no circle is found, it raises `DimeNotFoundError`.

It works in two stages, because the scale of every shape depends on this circle being accurate:

1. **Coarse search.** The scan is shrunk to 1,000 pixels wide and searched for the strongest circle (OpenCV's Hough circle transform) with a radius of 1.5% to 10% of the width.
2. **Refinement.** On the full-size scan, 360 rays are cast out from the coarse center. On each ray, the dime's edge is the outermost strong dark-to-bright change near the coarse radius. A circle is fit to those edge points by least squares, rays that disagree with it (such as those that hit the nugget) are dropped, and the circle is fit again. This repeats around its own result three times. If fewer than half the rays agree, the coarse circle is used.

The refinement exists for two reasons found while testing on the real scans. A plain circle search tended to lock onto the inner edge of the dime's dark rim, which makes the radius 3 to 5% too small. And the coarse center can be off by more than 10 pixels, which puts the true edge outside the search on one side. The outer edge is the dime's real boundary, so that's what is measured.

### NuggetFinder (`nugget_finder.py`)

`NuggetFinder.find(image, dime_detection)` returns a `NuggetDetection` with `contour_px`, the outline as an array of (x, y) pixel points, and `warnings`, a list of messages. If nothing is found, it raises `NuggetNotFoundError`.

The nugget is the largest region of strong color:

1. Take the saturation channel and blur it slightly, by 1% of the dime's radius.
2. Threshold it. The threshold is chosen automatically (Otsu's method) but is never below 40 out of 255, so that a scan with no nugget does not turn paper noise into one.
3. Blank out the dime and everything within 1.2 dime radii of its center.
4. Remove specks (crumbs) and close tiny gaps, using a filter 4% of the dime's radius across.
5. Take the outline of each region and keep the one with the largest area.

It warns, instead of failing, when the result may be wrong: a second region at least 5% of the nugget's size, or a nugget that touches the edge of the scan. A warning is a prompt to check the overlay by eye.

**Why saturation?** A nugget's brightness is easy to confuse with the shadows around it, which are about as dark as its darker parts. Color is different. The golden-brown nugget is strongly saturated, with values around 200 out of 255 on these scans, while the white paper, the silver dime and every shadow are nearly colorless, at roughly 0 to 30. Thresholding on saturation therefore ignores the shadows completely, and the automatic threshold lands around 90 to 100, well between the two.

### NuggetShape (`nugget_shape.py`)

`NuggetShape` is a dataclass holding the shape's `name` and its `polygon_mm`, a list of (x, y) points in millimeters. It provides `area_mm2`, `bounding_size_mm`, `to_json_dict()` and `NuggetShape.from_json_dict(name, json_dict)`.

It also builds shapes from scans, in two steps that `NuggetShape.from_pixel_contour` runs together:

- **`simplify_contour`** reduces the thousands of outline points to a polygon with about 50 to 60, using the Douglas-Peucker algorithm, which keeps points that lie on the outline and drops the ones that are within the tolerance of a straight line between their neighbors. The result stays in scan pixels, so it can be drawn on the scan.
- **`from_pixel_polygon`** converts that polygon to millimeters with the scale, moves it so its centroid is at (0, 0), and rounds each coordinate to 2 decimals. Every shape runs the same way around, clockwise as seen on screen, whichever way the scan traced it.

The overlay is drawn from the pixel polygon that these two steps share, so it shows exactly what is saved, up to the rounding.

### OverlayRenderer (`overlay_renderer.py`)

`OverlayRenderer` draws what was found so it can be checked by eye. `render(image, dime_detection, polygon_px)` crops the scan to the nugget and dime plus a margin, shrinks it to at most 1,400 pixels, and draws the dime's circle in blue and the polygon in green with a dot on each point. It never changes the scan itself. `save(overlay, path)` writes a JPEG, and `show(overlay, title)` opens a window that closes on any key press or when the window itself is closed. The overlays are committed with the shapes, which is why they are small, about 130 to 160 KB each.

### NuggetExtractor (`nugget_extractor.py`)

`NuggetExtractor` runs the steps above. `extract(scan_path)` does all the work without saving anything and returns an `ExtractionResult` with the shape, overlay, warnings and scale. `run(scan_path, show_overlay=True)` calls it, prints a summary and any warnings, saves the shape and overlay, and shows the overlay. The scan is loaded with OpenCV, which applies the rotation a phone stores with the photo. A missing or unreadable scan raises `ScanLoadError`.

The command line takes the scan path and an optional `--no-show`.

## Shape File Format

A shape file holds only the polygon, in millimeters:

```json
{
  "polygon": [
    [-12.42, -29.62],
    [-11.5, -30.42],
    [-9.48, -29.68],
    [1.27, -33.47]
  ]
}
```

This is an excerpt of `trex.json`, which has 53 points. Every point is on its own line, so a changed shape shows up clearly in a diff.

- **Units:** millimeters, as real-life size.
- **Origin:** the centroid of the shape, so a shape can be moved or rotated about its center.
- **Axes:** x to the right and y pointing **down**, as in the scan and as on a screen. A mirrored shape is therefore a real mirror image.
- **Order:** the points run clockwise as seen on screen, and the polygon is closed, so the last point joins the first.

The file holds no name, scale or source, since the name is the file name and the scale is already applied. A shape's source scan has the same name in `data/scans/`, and its check picture has the same name in `data/overlays/`.

## Settings

| Constant | File | Value | Meaning |
| --- | --- | --- | --- |
| `DIME_DIAMETER_MM` | `dime_detector.py` | 17.91 | The dime's real diameter, the size reference |
| `SIMPLIFICATION_TOLERANCE_MM` | `nugget_extractor.py` | 0.5 | How far the saved polygon may stray from the true outline |
| `MIN_NUGGET_SATURATION` | `nugget_finder.py` | 40 | The lowest saturation threshold allowed |
| `DIME_MASK_RADII` | `nugget_finder.py` | 1.2 | How far from the dime's center is ignored, in dime radii |
| `WARN_REGION_FRACTION` | `nugget_finder.py` | 0.05 | The size of a second region, as a share of the nugget, that triggers a warning |
| `OVERLAY_MAX_SIZE_PX` | `overlay_renderer.py` | 1400 | The largest side of a saved overlay |

There are more constants at the top of each file, with comments, for the smoothing, circle search and window.

**Why 0.5 mm?** Simplifying trades detail for points.

At 0.5 mm the polygon still keeps every arm, notch and tail, with about 55 points. Beyond about 1 mm, small features start to be cut off. Fewer points also make the packer faster.

## Checking a Result

Look at the overlay. The blue circle should sit on the dime's outer edge, and the green polygon should follow the nugget's outline all the way round, including its notches.

| Message | Meaning |
| --- | --- |
| `Warning: A second region is N% the size of the nugget...` | A second colorful region was found, e.g. a piece broken off or a second nugget. Check that the right region was picked. |
| `Warning: The nugget touches the edge of the scan...` | Part of the nugget may be cut off. Rescan with the nugget further from the edge. |
| `Cannot extract: Scan not found` / `Could not read an image` | The path is wrong or the file is not an image. |
| `Cannot extract: No dime found...` | No circle of the right size was found. Check the whole dime is in the scan. |
| `Cannot extract: No nugget found...` | No strongly colored region was found. Check that the nugget is in the scan and has its golden-brown color. |

## Tests

The tests are in `tests/` and use pytest. From the repository root:

```
.venv\Scripts\python -m pip install -r tools\extract\requirements-dev.txt
.venv\Scripts\python -m pytest
```

No test needs a real scan. `tests/mock_scans.py` draws mock scans, with white paper, a silver dime with a dark rim and golden-brown ellipses standing in for nuggets, so every size is known exactly and results can be compared against it.
