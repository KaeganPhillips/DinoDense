###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# dime_detector.py
#
# Finds the dime in a scan. The dime lies next to the nugget and is the size
# reference that turns pixels into millimeters.
###############################################################################

from dataclasses import dataclass

import cv2
import numpy as np

#------------------------------------------------------------------------------
# the coarse search runs on a copy of the scan shrunk to this width in pixels,
# so the circle search stays fast on very large scans
COARSE_IMAGE_WIDTH = 1000

# the dime radius as a fraction of the scan width, as a wide range that holds
# for any reasonable scan distance
MIN_RADIUS_FRACTION = 0.015
MAX_RADIUS_FRACTION = 0.10

# circle search settings, see cv2.HoughCircles
EDGE_THRESHOLD = 100
VOTE_THRESHOLD = 40

# the refined search looks in a square this many radii around the coarse
# circle, and blurs it by this fraction of the coarse radius
REFINE_CROP_RADII = 1.5
REFINE_BLUR_FRACTION = 0.02

# the dime edge is searched along this many rays from the coarse center, within
# this fraction of the coarse radius, in steps of this many pixels
REFINE_RAY_COUNT = 360
REFINE_RADIUS_TOLERANCE = 0.15
REFINE_STEP_PX = 0.5

# the refined search repeats around its own result, since the coarse center can
# be off by enough to put the true edge outside the search of some rays
REFINE_PASS_COUNT = 3

# a ray needs at least this much brightness change to hold the dime edge, so
# rays that leave the scan are ignored
MIN_EDGE_CONTRAST = 20

# the dime edge on a ray is the outermost dark to bright peak at least this
# share as strong as the strongest one, so that shading on the dime face, which
# can be stronger than the edge itself, is not mistaken for the edge
EDGE_STRENGTH_FRACTION = 0.5

# rays whose edge lies further than this fraction of the radius from the fitted
# circle are ignored, e.g. rays that hit the nugget or a shadow, and the circle
# is fit again this many times without them
EDGE_OUTLIER_FRACTION = 0.03
INLIER_FIT_COUNT = 4

# the refined circle is only trusted if at least this share of rays agree
MIN_INLIER_FRACTION = 0.5

#------------------------------------------------------------------------------
class DimeNotFoundError(Exception):
    '''Raised when no dime shaped circle can be found in a scan.'''
    pass

#------------------------------------------------------------------------------
@dataclass
class DimeDetection:
    '''The position and size of the dime found in a scan, in pixels.'''
    center_x_px: float
    center_y_px: float
    radius_px: float

    #--------------------------------------------------------------------------
    @property
    def diameter_px(self):
        '''The diameter of the dime in pixels.'''
        return 2 * self.radius_px

#------------------------------------------------------------------------------
class DimeDetector:
    '''
    Finds the most circular dime sized object in a scan. A coarse search on a
    shrunken copy of the scan finds the dime quickly, then repeated searches on
    the full size scan pin down its edge precisely, since the scale of every
    shape depends on it.
    '''

    #--------------------------------------------------------------------------
    def detect(self, image):
        '''
        Method to find the dime in a scan.

        param: image: the scan as a BGR image array, as loaded by cv2.imread
        return: a DimeDetection in the pixel coordinates of the full size scan
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        coarse_detection = self._find_coarse_circle(gray)

        detection = coarse_detection

        for _ in range(REFINE_PASS_COUNT):
            detection = self._refine_circle(gray, detection)

        return detection

    #--------------------------------------------------------------------------
    def _find_coarse_circle(self, gray):
        '''
        Method to find the dime on a shrunken copy of the scan.

        param: gray: the scan as a grayscale image array
        return: a DimeDetection in the pixel coordinates of the full size scan
        '''
        scale = COARSE_IMAGE_WIDTH / gray.shape[1]

        small = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        small = cv2.medianBlur(small, 5)

        circles = cv2.HoughCircles(
            small,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=COARSE_IMAGE_WIDTH,  # only the single best circle matters
            param1=EDGE_THRESHOLD,
            param2=VOTE_THRESHOLD,
            minRadius=round(MIN_RADIUS_FRACTION * COARSE_IMAGE_WIDTH),
            maxRadius=round(MAX_RADIUS_FRACTION * COARSE_IMAGE_WIDTH),
        )

        if circles is None:
            raise DimeNotFoundError(
                "No dime found. Check that a dime lies fully in the scan."
            )

        center_x, center_y, radius = circles[0][0]

        return DimeDetection(
            center_x_px=float(center_x / scale),
            center_y_px=float(center_y / scale),
            radius_px=float(radius / scale),
        )

    #--------------------------------------------------------------------------
    def _refine_circle(self, gray, coarse_detection):
        '''
        Method to pin down the dime edge on the full size scan. Rays are cast
        from the coarse center, and on each the outer edge of the dime is the
        strongest dark to bright change near the coarse radius. A circle is
        then fit to the edge points that agree. The outer edge is used since a
        circle search can lock onto the inner edge of the dime's dark rim. The
        coarse circle is returned if too few rays find an edge.

        param: gray: the scan as a grayscale image array
        param: coarse_detection: the DimeDetection from the coarse search
        return: a DimeDetection in the pixel coordinates of the full size scan
        '''
        coarse_radius = coarse_detection.radius_px
        margin = round(coarse_radius * REFINE_CROP_RADII)

        # clamp the crop to the scan
        left = max(0, round(coarse_detection.center_x_px) - margin)
        top = max(0, round(coarse_detection.center_y_px) - margin)
        right = min(
            gray.shape[1], round(coarse_detection.center_x_px) + margin
        )
        bottom = min(
            gray.shape[0], round(coarse_detection.center_y_px) + margin
        )

        crop = gray[top:bottom, left:right]
        crop = cv2.GaussianBlur(
            crop, (0, 0), sigmaX=coarse_radius * REFINE_BLUR_FRACTION
        ).astype(np.float32)

        # sample the brightness along every ray, outward from the coarse center
        angles = np.linspace(0, 2 * np.pi, REFINE_RAY_COUNT, endpoint=False)
        radii = np.arange(
            coarse_radius * (1 - REFINE_RADIUS_TOLERANCE),
            coarse_radius * (1 + REFINE_RADIUS_TOLERANCE),
            REFINE_STEP_PX,
        )
        center_x = coarse_detection.center_x_px - left
        center_y = coarse_detection.center_y_px - top

        sample_x = center_x + np.outer(np.cos(angles), radii)
        sample_y = center_y + np.outer(np.sin(angles), radii)
        profiles = cv2.remap(
            crop,
            sample_x.astype(np.float32),
            sample_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # the outer edge of the dime is the outermost strong peak of the dark
        # to bright change along each ray
        changes = np.diff(profiles, axis=1)
        is_peak = np.zeros(changes.shape, dtype=bool)
        is_peak[:, 1:-1] = (
            (changes[:, 1:-1] > changes[:, :-2])
            & (changes[:, 1:-1] >= changes[:, 2:])
        )
        is_edge = is_peak & (
            changes >= EDGE_STRENGTH_FRACTION * changes.max(axis=1,
                                                            keepdims=True)
        )
        edge_indices = (
            changes.shape[1] - 1 - np.argmax(is_edge[:, ::-1], axis=1)
        )
        edge_radii = radii[edge_indices] + REFINE_STEP_PX / 2

        contrast = profiles.max(axis=1) - profiles.min(axis=1)
        has_edge = contrast >= MIN_EDGE_CONTRAST

        ray_numbers = np.arange(REFINE_RAY_COUNT)
        edge_x = sample_x[ray_numbers, edge_indices]
        edge_y = sample_y[ray_numbers, edge_indices]

        # fit a circle to the edges, drop the rays that disagree with it, and
        # fit again, so rays that hit shading on the dime face or the nugget
        # do not pull the circle off the dime
        is_inlier = has_edge

        for _ in range(INLIER_FIT_COUNT):
            if is_inlier.sum() < MIN_INLIER_FRACTION * REFINE_RAY_COUNT:
                return coarse_detection

            fit_x, fit_y, fit_radius = self._fit_circle(
                edge_x[is_inlier], edge_y[is_inlier]
            )

            distances = np.hypot(edge_x - fit_x, edge_y - fit_y)
            is_inlier = has_edge & (
                np.abs(distances - fit_radius)
                <= EDGE_OUTLIER_FRACTION * fit_radius
            )

        return DimeDetection(
            center_x_px=float(fit_x + left),
            center_y_px=float(fit_y + top),
            radius_px=float(fit_radius),
        )

    #--------------------------------------------------------------------------
    @staticmethod
    def _fit_circle(points_x, points_y):
        '''
        Method to fit a circle to points by least squares.

        param: points_x: the x coordinates of the points
        param: points_y: the y coordinates of the points
        return: a tuple of the circle center x, center y, and radius
        '''
        design = np.column_stack([2 * points_x, 2 * points_y,
                                  np.ones_like(points_x)])
        target = points_x ** 2 + points_y ** 2

        (center_x, center_y, offset), *_ = np.linalg.lstsq(
            design, target, rcond=None
        )

        radius = np.sqrt(offset + center_x ** 2 + center_y ** 2)

        return center_x, center_y, radius

