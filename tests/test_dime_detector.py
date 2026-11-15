###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# test_dime_detector.py
#
# Tests for the dime detector. Scans are drawn as mock images with a known
# dime, so no test needs a real scan.
###############################################################################

import cv2
import numpy as np
import pytest

from dime_detector import DimeDetection, DimeDetector, DimeNotFoundError

# how close the detected dime must be to the drawn dime, in pixels
CENTER_TOLERANCE = 2
RADIUS_TOLERANCE = 2

#------------------------------------------------------------------------------
def draw_mock_scan(width, height, dime_x, dime_y, dime_radius, with_nugget):
    '''
    Method to draw a mock scan: white paper with a silver dime that has a dark
    rim and, optionally, a golden-brown blob standing in for the nugget.

    param: width: the scan width in pixels
    param: height: the scan height in pixels
    param: dime_x: the dime center x in pixels
    param: dime_y: the dime center y in pixels
    param: dime_radius: the dime radius in pixels
    param: with_nugget: True to draw a nugget below the dime
    return: the scan as a BGR image array
    '''
    scan = np.full((height, width, 3), 250, dtype=np.uint8)

    # the dark rim is drawn first, so the dime ends exactly at dime_radius
    rim_width = max(2, dime_radius // 25)
    cv2.circle(scan, (dime_x, dime_y), dime_radius, (90, 90, 90), -1)
    cv2.circle(scan, (dime_x, dime_y), dime_radius - rim_width,
               (200, 200, 200), -1)

    if with_nugget:
        nugget_center = (dime_x, dime_y + 4 * dime_radius)
        axes = (2 * dime_radius, 3 * dime_radius)
        cv2.ellipse(scan, nugget_center, axes, 20, 0, 360, (40, 140, 220), -1)

    # soften the edges the way a camera does
    return cv2.GaussianBlur(scan, (0, 0), sigmaX=1.5)

#------------------------------------------------------------------------------
def test_diameter_is_twice_the_radius():
    detection = DimeDetection(center_x_px=10, center_y_px=20, radius_px=7.5)

    assert detection.diameter_px == 15

#------------------------------------------------------------------------------
def test_detects_dime_in_scan_without_nugget():
    scan = draw_mock_scan(1000, 1200, 400, 300, 50, with_nugget=False)

    detection = DimeDetector().detect(scan)

    assert detection.center_x_px == pytest.approx(400, abs=CENTER_TOLERANCE)
    assert detection.center_y_px == pytest.approx(300, abs=CENTER_TOLERANCE)
    assert detection.radius_px == pytest.approx(50, abs=RADIUS_TOLERANCE)

#------------------------------------------------------------------------------
def test_detects_dime_next_to_nugget():
    scan = draw_mock_scan(1000, 1200, 400, 300, 50, with_nugget=True)

    detection = DimeDetector().detect(scan)

    assert detection.center_x_px == pytest.approx(400, abs=CENTER_TOLERANCE)
    assert detection.center_y_px == pytest.approx(300, abs=CENTER_TOLERANCE)
    assert detection.radius_px == pytest.approx(50, abs=RADIUS_TOLERANCE)

#------------------------------------------------------------------------------
def test_detects_dime_in_scan_larger_than_the_coarse_image():
    scan = draw_mock_scan(4000, 5000, 1700, 1300, 160, with_nugget=True)

    detection = DimeDetector().detect(scan)

    assert detection.center_x_px == pytest.approx(1700, abs=CENTER_TOLERANCE)
    assert detection.center_y_px == pytest.approx(1300, abs=CENTER_TOLERANCE)
    assert detection.radius_px == pytest.approx(160, abs=RADIUS_TOLERANCE)

#------------------------------------------------------------------------------
def test_detects_dime_near_the_scan_edge():
    scan = draw_mock_scan(1000, 1200, 60, 70, 50, with_nugget=False)

    detection = DimeDetector().detect(scan)

    assert detection.center_x_px == pytest.approx(60, abs=CENTER_TOLERANCE)
    assert detection.center_y_px == pytest.approx(70, abs=CENTER_TOLERANCE)
    assert detection.radius_px == pytest.approx(50, abs=RADIUS_TOLERANCE)

#------------------------------------------------------------------------------
def test_raises_when_there_is_no_dime():
    scan = np.full((1200, 1000, 3), 250, dtype=np.uint8)

    with pytest.raises(DimeNotFoundError):
        DimeDetector().detect(scan)

