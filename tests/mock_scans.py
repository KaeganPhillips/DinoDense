###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# mock_scans.py
#
# Draws mock scans for the tests: white paper with a silver dime that has a
# dark rim and, optionally, golden-brown ellipses standing in for nuggets.
# Every size is known exactly, so tests can compare against it.
###############################################################################

import cv2
import numpy as np

#------------------------------------------------------------------------------
PAPER_COLOR = (250, 250, 250)
DIME_COLOR = (200, 200, 200)
DIME_RIM_COLOR = (90, 90, 90)
NUGGET_COLOR = (40, 140, 220)  # golden brown, as BGR

# how the default nugget sits below the dime, in dime radii
NUGGET_OFFSET_RADII = 4
NUGGET_HALF_WIDTH_RADII = 2
NUGGET_HALF_HEIGHT_RADII = 3
NUGGET_ANGLE_DEGREES = 20

#------------------------------------------------------------------------------
def make_blank_scan(width, height):
    '''
    Method to make a blank scan of white paper.

    param: width: the scan width in pixels
    param: height: the scan height in pixels
    return: the scan as a BGR image array
    '''
    return np.full((height, width, 3), PAPER_COLOR, dtype=np.uint8)

#------------------------------------------------------------------------------
def draw_dime(scan, dime_x, dime_y, dime_radius):
    '''
    Method to draw a silver dime with a dark rim onto a scan. The dime ends
    exactly at dime_radius.

    param: scan: the scan to draw on, changed in place
    param: dime_x: the dime center x in pixels
    param: dime_y: the dime center y in pixels
    param: dime_radius: the dime radius in pixels
    '''
    # the dark rim is drawn first, so the dime ends exactly at dime_radius
    rim_width = max(2, dime_radius // 25)
    cv2.circle(scan, (dime_x, dime_y), dime_radius, DIME_RIM_COLOR, -1)
    cv2.circle(scan, (dime_x, dime_y), dime_radius - rim_width,
               DIME_COLOR, -1)

#------------------------------------------------------------------------------
def draw_nugget(scan, center_x, center_y, half_width, half_height,
                angle_degrees=NUGGET_ANGLE_DEGREES):
    '''
    Method to draw a golden-brown ellipse standing in for a nugget.

    param: scan: the scan to draw on, changed in place
    param: center_x: the ellipse center x in pixels
    param: center_y: the ellipse center y in pixels
    param: half_width: half the ellipse width in pixels
    param: half_height: half the ellipse height in pixels
    param: angle_degrees: how far the ellipse is rotated
    '''
    cv2.ellipse(scan, (center_x, center_y), (half_width, half_height),
                angle_degrees, 0, 360, NUGGET_COLOR, -1)

#------------------------------------------------------------------------------
def soften(scan):
    '''
    Method to soften the edges of a scan the way a camera does.

    param: scan: the scan to soften
    return: the softened scan
    '''
    return cv2.GaussianBlur(scan, (0, 0), sigmaX=1.5)

#------------------------------------------------------------------------------
def draw_mock_scan(width, height, dime_x, dime_y, dime_radius, with_nugget):
    '''
    Method to draw a mock scan: white paper with a dime and, optionally, one
    nugget below the dime.

    param: width: the scan width in pixels
    param: height: the scan height in pixels
    param: dime_x: the dime center x in pixels
    param: dime_y: the dime center y in pixels
    param: dime_radius: the dime radius in pixels
    param: with_nugget: True to draw a nugget below the dime
    return: the scan as a BGR image array
    '''
    scan = make_blank_scan(width, height)
    draw_dime(scan, dime_x, dime_y, dime_radius)

    if with_nugget:
        draw_nugget(
            scan,
            dime_x,
            dime_y + NUGGET_OFFSET_RADII * dime_radius,
            NUGGET_HALF_WIDTH_RADII * dime_radius,
            NUGGET_HALF_HEIGHT_RADII * dime_radius,
        )

    return soften(scan)

