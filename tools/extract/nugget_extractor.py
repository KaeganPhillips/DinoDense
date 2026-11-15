###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# nugget_extractor.py
#
# Extracts a dino-nugget shape from a scan of one nugget lying next to a dime
# on white paper. The dime gives the scale, the nugget outline becomes a
# polygon in millimeters saved to data/shapes/, and an overlay of what was
# found is saved to data/overlays/ and shown, so it can be checked by eye.
# The name of the shape is the name of the scan file.
#
# Usage: python tools/extract/nugget_extractor.py data/scans/trex.jpg
#        [--no-show]
###############################################################################

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

from dime_detector import DimeDetector, DimeNotFoundError
from nugget_finder import NuggetFinder, NuggetNotFoundError
from nugget_shape import NuggetShape
from overlay_renderer import OverlayRenderer

#------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAPES_DIR = PROJECT_ROOT / "data" / "shapes"
OVERLAYS_DIR = PROJECT_ROOT / "data" / "overlays"

# how far in millimeters the saved polygon may stray from the nugget outline
SIMPLIFICATION_TOLERANCE_MM = 0.5

#------------------------------------------------------------------------------
class ScanLoadError(Exception):
    '''Raised when a scan file is missing or is not an image.'''
    pass

#------------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    '''
    Everything found in one scan.

    shape: the NuggetShape that is saved
    overlay: the overlay image of the dime and the shape drawn on the scan
    warnings: a list of messages about anything that may have gone wrong
    mm_per_px: the millimeters one pixel of the scan covers
    '''
    shape: NuggetShape
    overlay: object
    warnings: list
    mm_per_px: float

#------------------------------------------------------------------------------
class NuggetExtractor:
    '''
    Extracts nugget shapes from scans. Finds the dime, works out the scale from
    it, finds the nugget, and turns its outline into a NuggetShape.
    '''

    #--------------------------------------------------------------------------
    def __init__(self, shapes_dir=None, overlays_dir=None):
        '''
        param: shapes_dir: the folder shapes are saved to (default =
               data/shapes)
        param: overlays_dir: the folder overlays are saved to (default =
               data/overlays)
        '''
        self.shapes_dir = Path(shapes_dir or SHAPES_DIR)
        self.overlays_dir = Path(overlays_dir or OVERLAYS_DIR)

        self.dime_detector = DimeDetector()
        self.nugget_finder = NuggetFinder()
        self.overlay_renderer = OverlayRenderer()

    #--------------------------------------------------------------------------
    def extract(self, scan_path):
        '''
        Method to extract the shape from a scan, without saving anything.

        param: scan_path: the file path of the scan, e.g. data/scans/trex.jpg
        return: an ExtractionResult
        '''
        scan_path = Path(scan_path)
        image = self._load_scan(scan_path)

        dime_detection = self.dime_detector.detect(image)
        mm_per_px = dime_detection.mm_per_px

        nugget_detection = self.nugget_finder.find(image, dime_detection)

        # the polygon stays in pixels to be drawn, and is converted to
        # millimeters to be saved, so the overlay shows what is saved
        polygon_px = NuggetShape.simplify_contour(
            nugget_detection.contour_px, mm_per_px, SIMPLIFICATION_TOLERANCE_MM
        )
        shape = NuggetShape.from_pixel_polygon(
            scan_path.stem, polygon_px, mm_per_px
        )

        overlay = self.overlay_renderer.render(
            image, dime_detection, polygon_px
        )

        return ExtractionResult(
            shape=shape,
            overlay=overlay,
            warnings=nugget_detection.warnings,
            mm_per_px=mm_per_px,
        )

    #--------------------------------------------------------------------------
    def run(self, scan_path, show_overlay=True):
        '''
        Method to extract the shape from a scan, save the shape and the
        overlay, and show the overlay. Files from an earlier run are
        overwritten.

        param: scan_path: the file path of the scan, e.g. data/scans/trex.jpg
        param: show_overlay: True to show the overlay in a window until it is
               closed
        return: the ExtractionResult
        '''
        result = self.extract(scan_path)
        name = result.shape.name

        width, height = result.shape.bounding_size_mm
        print(f"Scale: {result.mm_per_px:.4f} mm per pixel (from the dime).")
        print(
            f"Found {name}: {len(result.shape.polygon_mm)} points, "
            f"{width:.1f} x {height:.1f} mm, "
            f"{result.shape.area_mm2:.0f} mm2."
        )

        for warning in result.warnings:
            print(f"Warning: {warning}")

        shape_path = self.shapes_dir / f"{name}.json"
        self.save_shape(result.shape, shape_path)
        print(f"Saved shape to {shape_path}")

        overlay_path = self.overlays_dir / f"{name}.jpg"
        self.overlay_renderer.save(result.overlay, overlay_path)
        print(f"Saved overlay to {overlay_path}")

        if show_overlay:
            self.overlay_renderer.show(
                result.overlay, f"{name} - press any key to close"
            )

        return result

    #--------------------------------------------------------------------------
    @staticmethod
    def save_shape(shape, path):
        '''
        Method to save a shape as a JSON file, overwriting any existing file
        and creating the folder if needed. Every point has its own line, so a
        changed shape shows up clearly in a diff.

        param: shape: the NuggetShape to save
        param: path: the file path to save the shape to
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)

        point_lines = ",\n".join(
            f"    {json.dumps(point)}"
            for point in shape.to_json_dict()["polygon"]
        )

        with open(path, "w", encoding="utf-8") as shape_file:
            shape_file.write(f'{{\n  "polygon": [\n{point_lines}\n  ]\n}}\n')

    #--------------------------------------------------------------------------
    @staticmethod
    def _load_scan(scan_path):
        '''
        Method to load a scan. The phone's rotation of the photo is applied.

        param: scan_path: the file path of the scan
        return: the scan as a BGR image array
        '''
        if not scan_path.is_file():
            raise ScanLoadError(f"Scan not found: {scan_path}")

        image = cv2.imread(str(scan_path))

        if image is None:
            raise ScanLoadError(f"Could not read an image from {scan_path}")

        return image

#------------------------------------------------------------------------------
def main():
    '''
    Method to extract the shape from one scan, save it and its overlay, and
    show the overlay.
    '''
    parser = argparse.ArgumentParser(
        description="Extract a dino-nugget shape from a scan."
    )
    parser.add_argument(
        "scan_path",
        help="Path of the scan, e.g. data/scans/trex.jpg. The scan file name "
             "is the name of the shape."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the overlay without showing it in a window."
    )
    args = parser.parse_args()

    try:
        NuggetExtractor().run(args.scan_path, show_overlay=not args.no_show)

    except (ScanLoadError, DimeNotFoundError, NuggetNotFoundError) as e:
        print(f"Cannot extract: {e}")
        sys.exit(1)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

