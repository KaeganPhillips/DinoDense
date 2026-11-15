###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# nugget_finder.py
#
# Finds the outline of a dino-nugget in a scan. The golden-brown nugget is far
# more colorful than the white paper, the silver dime, and their shadows, so
# the nugget is the largest region of strongly saturated color.
###############################################################################

from dataclasses import dataclass, field

import cv2
import numpy as np

#------------------------------------------------------------------------------
# the saturation the nugget must at least have, from 0 to 255. The threshold is
# normally picked automatically, but never lower than this, so a scan with no
# nugget does not turn paper noise into one.
MIN_NUGGET_SATURATION = 40

# the scan is blurred by this fraction of the dime radius before thresholding,
# to smooth away jpeg noise
BLUR_FRACTION = 0.01

# the dime and its surroundings are ignored out to this many dime radii
DIME_MASK_RADII = 1.2

# specks and gaps smaller than this fraction of the dime radius are cleaned up
CLEANUP_FRACTION = 0.04

# a second region at least this fraction of the nugget's area gets a warning,
# since it may be a second nugget or a large piece of one
WARN_REGION_FRACTION = 0.05

#------------------------------------------------------------------------------
class NuggetNotFoundError(Exception):
    '''Raised when no nugget can be found in a scan.'''
    pass

#------------------------------------------------------------------------------
@dataclass
class NuggetDetection:
    '''
    The outline of the nugget found in a scan.

    contour_px: the outline as an array of (x, y) points in pixels
    warnings: a list of messages about anything that may have gone wrong
    '''
    contour_px: np.ndarray
    warnings: list = field(default_factory=list)

#------------------------------------------------------------------------------
class NuggetFinder:
    '''
    Finds the outline of the nugget in a scan, given where the dime is. The
    dime is ignored, and its radius sets the size of the smoothing so the
    finder behaves the same on scans of any resolution.
    '''

    #--------------------------------------------------------------------------
    def find(self, image, dime_detection):
        '''
        Method to find the nugget outline. The largest colorful region is the
        nugget. Any problem that leaves the result in doubt is reported as a
        warning instead of an error, so the overlay can be checked by eye.

        param: image: the scan as a BGR image array, as loaded by cv2.imread
        param: dime_detection: the DimeDetection of the dime in the scan
        return: a NuggetDetection with the outline in the pixel coordinates of
                the scan
        '''
        mask = self._make_nugget_mask(image, dime_detection)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            raise NuggetNotFoundError(
                "No nugget found. Check that the nugget is fully in the scan "
                "and has a golden-brown color."
            )

        # the nugget is the region with the largest area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        nugget_contour = contours[0]
        nugget_area = cv2.contourArea(nugget_contour)

        warnings = []

        # a second large region may mean the wrong region was picked
        if len(contours) > 1:
            other_area = cv2.contourArea(contours[1])

            if other_area >= WARN_REGION_FRACTION * nugget_area:
                warnings.append(
                    f"A second region is {other_area / nugget_area:.0%} the "
                    f"size of the nugget. Check that the right one was picked."
                )

        if self._touches_edge(nugget_contour, image.shape):
            warnings.append(
                "The nugget touches the edge of the scan, so part of it may "
                "be cut off."
            )

        return NuggetDetection(
            contour_px=nugget_contour.reshape(-1, 2).astype(np.float64),
            warnings=warnings,
        )

    #--------------------------------------------------------------------------
    def _make_nugget_mask(self, image, dime_detection):
        '''
        Method to make a mask that is 255 on colorful pixels and 0 elsewhere,
        ignoring the dime and cleaning up specks.

        param: image: the scan as a BGR image array
        param: dime_detection: the DimeDetection of the dime in the scan
        return: the mask as a uint8 image array
        '''
        dime_radius = dime_detection.radius_px

        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
        saturation = cv2.GaussianBlur(
            saturation, (0, 0), sigmaX=dime_radius * BLUR_FRACTION
        )

        # paper is nearly colorless and the nugget is strongly colored, so
        # the automatic threshold falls between the two
        otsu_threshold, _ = cv2.threshold(
            saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = max(otsu_threshold, MIN_NUGGET_SATURATION)

        mask = np.where(saturation > threshold, 255, 0).astype(np.uint8)

        # ignore the dime, whose engraving can have colorful pixels
        cv2.circle(
            mask,
            (round(dime_detection.center_x_px),
             round(dime_detection.center_y_px)),
            round(dime_radius * DIME_MASK_RADII),
            0,
            -1,
        )

        # remove specks (crumbs), then close tiny gaps in the nugget
        kernel_size = max(3, round(dime_radius * CLEANUP_FRACTION)) | 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    #--------------------------------------------------------------------------
    @staticmethod
    def _touches_edge(contour, image_shape):
        '''
        Method to check if a contour touches the edge of the scan.

        param: contour: the contour as returned by cv2.findContours
        param: image_shape: the shape of the scan image array
        return: True if the contour touches the edge of the scan
        '''
        x, y, width, height = cv2.boundingRect(contour)
        image_height, image_width = image_shape[:2]

        return (
            x == 0 or y == 0
            or x + width >= image_width or y + height >= image_height
        )

