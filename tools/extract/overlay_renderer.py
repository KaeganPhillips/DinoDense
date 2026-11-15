###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# overlay_renderer.py
#
# Draws what the extractor found on top of the scan, so that it can be checked
# by eye: the dime circle the scale came from, and the polygon that is saved
# as the nugget shape. Overlays are committed with the shapes, so they are kept
# small.
###############################################################################

import math
import os

import cv2
import numpy as np

#------------------------------------------------------------------------------
# the overlay is cropped to the nugget and dime, plus this fraction of the
# larger side of that area on every side
CROP_MARGIN_FRACTION = 0.10

# the overlay is shrunk until its larger side is at most this many pixels, and
# is never enlarged
OVERLAY_MAX_SIZE_PX = 1400

# the colors as BGR, picked to stand out on both the paper and the nugget
POLYGON_COLOR = (0, 220, 0)
DIME_COLOR = (255, 100, 0)

LINE_THICKNESS = 2
VERTEX_RADIUS = 3

JPEG_QUALITY = 90

# the window showing the overlay is shrunk to fit this height, so it fits on a
# screen
WINDOW_MAX_HEIGHT_PX = 900

#------------------------------------------------------------------------------
class OverlayRenderer:
    '''
    Draws the dime circle and the nugget polygon on the scan, then shows or
    saves the result.
    '''

    #--------------------------------------------------------------------------
    def render(self, image, dime_detection, polygon_px):
        '''
        Method to draw the overlay. The scan itself is not changed.

        param: image: the scan as a BGR image array, as loaded by cv2.imread
        param: dime_detection: the DimeDetection of the dime in the scan
        param: polygon_px: the nugget polygon as an array of (x, y) points in
               the pixel coordinates of the scan
        return: the overlay as a BGR image array, cropped to the nugget and
                dime and shrunk to a reasonable size
        '''
        polygon_px = np.asarray(polygon_px, dtype=np.float64)

        left, top, right, bottom = self._get_crop_box(
            image.shape, dime_detection, polygon_px
        )

        crop = image[top:bottom, left:right]
        scale = min(1.0, OVERLAY_MAX_SIZE_PX / max(crop.shape[:2]))

        if scale < 1.0:
            overlay = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            overlay = crop.copy()

        # draw after shrinking, so the lines are the same width on any scan
        cv2.circle(
            overlay,
            (round((dime_detection.center_x_px - left) * scale),
             round((dime_detection.center_y_px - top) * scale)),
            round(dime_detection.radius_px * scale),
            DIME_COLOR,
            LINE_THICKNESS,
            cv2.LINE_AA,
        )

        points = np.round(
            (polygon_px - [left, top]) * scale
        ).astype(np.int32)

        cv2.polylines(
            overlay, [points], True, POLYGON_COLOR, LINE_THICKNESS, cv2.LINE_AA
        )

        for x, y in points:
            cv2.circle(
                overlay, (int(x), int(y)), VERTEX_RADIUS, POLYGON_COLOR, -1,
                cv2.LINE_AA
            )

        return overlay

    #--------------------------------------------------------------------------
    def save(self, overlay, path):
        '''
        Method to save an overlay as a JPEG file, overwriting any existing
        file and creating the folder if needed.

        param: overlay: the overlay as a BGR image array
        param: path: the file path to save the overlay to
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)

        was_saved = cv2.imwrite(
            str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if not was_saved:
            raise OSError(f"Could not save the overlay to {path}")

    #--------------------------------------------------------------------------
    def show(self, overlay, title):
        '''
        Method to show an overlay in a window until any key is pressed or the
        window is closed.

        param: overlay: the overlay as a BGR image array
        param: title: the title of the window
        '''
        height, width = overlay.shape[:2]
        window_scale = min(1.0, WINDOW_MAX_HEIGHT_PX / height)

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            title, round(width * window_scale), round(height * window_scale)
        )
        cv2.imshow(title, overlay)

        # waitKey also lets the window draw, and the window is closed by hand
        # when its visibility drops below 1
        while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(50) != -1:
                break

        cv2.destroyWindow(title)

    #--------------------------------------------------------------------------
    @staticmethod
    def _get_crop_box(image_shape, dime_detection, polygon_px):
        '''
        Method to get the area of the scan the overlay is cropped to: the dime
        and the nugget with a margin, kept inside the scan.

        param: image_shape: the shape of the scan image array
        param: dime_detection: the DimeDetection of the dime in the scan
        param: polygon_px: the nugget polygon as an array of (x, y) points
        return: a tuple of the left, top, right, and bottom pixel of the area
        '''
        radius = dime_detection.radius_px
        dime_x = dime_detection.center_x_px
        dime_y = dime_detection.center_y_px

        x_min = min(polygon_px[:, 0].min(), dime_x - radius)
        x_max = max(polygon_px[:, 0].max(), dime_x + radius)
        y_min = min(polygon_px[:, 1].min(), dime_y - radius)
        y_max = max(polygon_px[:, 1].max(), dime_y + radius)

        margin = CROP_MARGIN_FRACTION * max(x_max - x_min, y_max - y_min)
        image_height, image_width = image_shape[:2]

        return (
            max(0, math.floor(x_min - margin)),
            max(0, math.floor(y_min - margin)),
            min(image_width, math.ceil(x_max + margin)),
            min(image_height, math.ceil(y_max + margin)),
        )

