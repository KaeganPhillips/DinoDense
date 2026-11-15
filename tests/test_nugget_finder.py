###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# test_nugget_finder.py
#
# Tests for the nugget finder. Scans are drawn as mock images with nuggets of
# a known size (see mock_scans.py), so no test needs a real scan.
###############################################################################

import cv2
import numpy as np
import pytest

from dime_detector import DimeDetection
from mock_scans import (
    NUGGET_HALF_HEIGHT_RADII,
    NUGGET_HALF_WIDTH_RADII,
    NUGGET_OFFSET_RADII,
    draw_dime,
    draw_mock_scan,
    draw_nugget,
    make_blank_scan,
    soften,
)
from nugget_finder import NuggetFinder, NuggetNotFoundError

WIDTH = 1000
HEIGHT = 1400
DIME_X = 400
DIME_Y = 300
DIME_RADIUS = 50

# how close the found nugget area must be to the drawn area, as a fraction
AREA_TOLERANCE = 0.02

#------------------------------------------------------------------------------
def make_dime_detection():
    '''
    Method to make the DimeDetection that matches the drawn dime exactly.

    return: a DimeDetection
    '''
    return DimeDetection(DIME_X, DIME_Y, DIME_RADIUS)

#------------------------------------------------------------------------------
def get_contour_area(detection):
    '''
    Method to get the area inside a found outline.

    param: detection: a NuggetDetection
    return: the area in square pixels
    '''
    return cv2.contourArea(detection.contour_px.astype(np.float32))

#------------------------------------------------------------------------------
def test_finds_nugget_area():
    scan = draw_mock_scan(WIDTH, HEIGHT, DIME_X, DIME_Y, DIME_RADIUS, True)

    detection = NuggetFinder().find(scan, make_dime_detection())

    drawn_area = np.pi * (
        NUGGET_HALF_WIDTH_RADII * DIME_RADIUS
        * NUGGET_HALF_HEIGHT_RADII * DIME_RADIUS
    )
    assert get_contour_area(detection) == pytest.approx(
        drawn_area, rel=AREA_TOLERANCE
    )
    assert detection.warnings == []

#------------------------------------------------------------------------------
def test_finds_nugget_position():
    scan = draw_mock_scan(WIDTH, HEIGHT, DIME_X, DIME_Y, DIME_RADIUS, True)

    detection = NuggetFinder().find(scan, make_dime_detection())

    center_x, center_y = detection.contour_px.mean(axis=0)
    assert center_x == pytest.approx(DIME_X, abs=5)
    assert center_y == pytest.approx(
        DIME_Y + NUGGET_OFFSET_RADII * DIME_RADIUS, abs=5
    )

#------------------------------------------------------------------------------
def test_outline_does_not_include_the_dime():
    scan = draw_mock_scan(WIDTH, HEIGHT, DIME_X, DIME_Y, DIME_RADIUS, True)

    detection = NuggetFinder().find(scan, make_dime_detection())

    distances = np.hypot(
        detection.contour_px[:, 0] - DIME_X,
        detection.contour_px[:, 1] - DIME_Y,
    )
    assert distances.min() > DIME_RADIUS

#------------------------------------------------------------------------------
def test_ignores_colorful_pixels_on_the_dime():
    scan = make_blank_scan(WIDTH, HEIGHT)
    draw_dime(scan, DIME_X, DIME_Y, DIME_RADIUS)
    draw_nugget(scan, DIME_X, DIME_Y + 4 * DIME_RADIUS, 100, 150)

    # a golden-brown speck on the dime, like a tinted engraving
    cv2.circle(scan, (DIME_X, DIME_Y), DIME_RADIUS // 3, (40, 140, 220), -1)

    detection = NuggetFinder().find(soften(scan), make_dime_detection())

    assert detection.contour_px[:, 1].min() > DIME_Y + DIME_RADIUS
    assert detection.warnings == []

#------------------------------------------------------------------------------
def test_ignores_crumbs():
    scan = make_blank_scan(WIDTH, HEIGHT)
    draw_dime(scan, DIME_X, DIME_Y, DIME_RADIUS)
    draw_nugget(scan, DIME_X, DIME_Y + 4 * DIME_RADIUS, 100, 150)

    # a few crumbs on the paper, far from the nugget
    for crumb_x, crumb_y in [(800, 200), (850, 900), (100, 1100)]:
        cv2.circle(scan, (crumb_x, crumb_y), 4, (40, 140, 220), -1)

    detection = NuggetFinder().find(soften(scan), make_dime_detection())

    drawn_area = np.pi * 100 * 150
    assert get_contour_area(detection) == pytest.approx(
        drawn_area, rel=AREA_TOLERANCE
    )
    assert detection.warnings == []

#------------------------------------------------------------------------------
def test_picks_the_largest_region_and_warns_about_a_second_one():
    scan = make_blank_scan(WIDTH, HEIGHT)
    draw_dime(scan, DIME_X, DIME_Y, DIME_RADIUS)
    draw_nugget(scan, DIME_X, DIME_Y + 4 * DIME_RADIUS, 100, 150)
    draw_nugget(scan, 800, 1000, 60, 80)

    detection = NuggetFinder().find(soften(scan), make_dime_detection())

    assert get_contour_area(detection) == pytest.approx(
        np.pi * 100 * 150, rel=AREA_TOLERANCE
    )
    assert len(detection.warnings) == 1
    assert "second region" in detection.warnings[0]

#------------------------------------------------------------------------------
def test_warns_when_the_nugget_touches_the_scan_edge():
    scan = make_blank_scan(WIDTH, HEIGHT)
    draw_dime(scan, DIME_X, DIME_Y, DIME_RADIUS)
    draw_nugget(scan, DIME_X, HEIGHT - 20, 100, 150)

    detection = NuggetFinder().find(soften(scan), make_dime_detection())

    assert len(detection.warnings) == 1
    assert "edge" in detection.warnings[0]

#------------------------------------------------------------------------------
def test_works_on_a_scan_with_a_different_resolution():
    scale = 4
    scan = draw_mock_scan(
        WIDTH * scale, HEIGHT * scale, DIME_X * scale, DIME_Y * scale,
        DIME_RADIUS * scale, True
    )
    dime_detection = DimeDetection(
        DIME_X * scale, DIME_Y * scale, DIME_RADIUS * scale
    )

    detection = NuggetFinder().find(scan, dime_detection)

    drawn_area = np.pi * (
        NUGGET_HALF_WIDTH_RADII * NUGGET_HALF_HEIGHT_RADII
        * (DIME_RADIUS * scale) ** 2
    )
    assert get_contour_area(detection) == pytest.approx(
        drawn_area, rel=AREA_TOLERANCE
    )

#------------------------------------------------------------------------------
def test_raises_when_there_is_no_nugget():
    scan = draw_mock_scan(WIDTH, HEIGHT, DIME_X, DIME_Y, DIME_RADIUS, False)

    with pytest.raises(NuggetNotFoundError):
        NuggetFinder().find(scan, make_dime_detection())

