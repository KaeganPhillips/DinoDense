###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# test_dime_detector.py
#
# Tests for the dime detector. Scans are drawn as mock images with a known
# dime (see mock_scans.py), so no test needs a real scan.
###############################################################################

import pytest

from dime_detector import DimeDetection, DimeDetector, DimeNotFoundError
from mock_scans import draw_mock_scan, make_blank_scan

# how close the detected dime must be to the drawn dime, in pixels
CENTER_TOLERANCE = 2
RADIUS_TOLERANCE = 2

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
    scan = make_blank_scan(1000, 1200)

    with pytest.raises(DimeNotFoundError):
        DimeDetector().detect(scan)

