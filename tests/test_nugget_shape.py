###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# test_nugget_shape.py
#
# Tests for the nugget shape: its measurements, its JSON form, and making one
# from an outline in pixels. Outlines are drawn from known shapes, so the right
# answers are known exactly.
###############################################################################

import numpy as np
import pytest

from nugget_shape import NuggetShape

MM_PER_PX = 0.1
TOLERANCE_MM = 0.3

#------------------------------------------------------------------------------
def make_rectangle_contour(left, top, width, height):
    '''
    Method to make the outline of a rectangle with a point on every pixel of
    its edge, the way a scan outline has.

    param: left: the left edge in pixels
    param: top: the top edge in pixels
    param: width: the width in pixels
    param: height: the height in pixels
    return: an array of (x, y) points running clockwise on screen
    '''
    right = left + width
    bottom = top + height

    points = (
        [(x, top) for x in range(left, right)]
        + [(right, y) for y in range(top, bottom)]
        + [(x, bottom) for x in range(right, left, -1)]
        + [(left, y) for y in range(bottom, top, -1)]
    )

    return np.array(points, dtype=np.float64)

#------------------------------------------------------------------------------
def make_circle_contour(center_x, center_y, radius, point_count=2000):
    '''
    Method to make the outline of a circle with many points.

    param: center_x: the circle center x in pixels
    param: center_y: the circle center y in pixels
    param: radius: the circle radius in pixels
    param: point_count: how many points the outline has
    return: an array of (x, y) points running clockwise on screen
    '''
    angles = np.linspace(0, 2 * np.pi, point_count, endpoint=False)

    return np.column_stack([
        center_x + radius * np.cos(angles),
        center_y + radius * np.sin(angles),
    ])

#------------------------------------------------------------------------------
def test_area_of_a_rectangle():
    shape = NuggetShape("box", [(0, 0), (30, 0), (30, 20), (0, 20)])

    assert shape.area_mm2 == pytest.approx(600)

#------------------------------------------------------------------------------
def test_area_does_not_depend_on_the_direction_of_the_points():
    clockwise = NuggetShape("box", [(0, 0), (30, 0), (30, 20), (0, 20)])
    counter_clockwise = NuggetShape("box", clockwise.polygon_mm[::-1])

    assert counter_clockwise.area_mm2 == pytest.approx(600)

#------------------------------------------------------------------------------
def test_bounding_size():
    shape = NuggetShape("triangle", [(-5, -2), (15, -2), (4, 12)])

    assert shape.bounding_size_mm == (20, 14)

#------------------------------------------------------------------------------
def test_json_round_trip():
    shape = NuggetShape("box", [(0, 0), (30.5, 0), (30.5, 20.25), (0, 20.25)])

    loaded = NuggetShape.from_json_dict("box", shape.to_json_dict())

    assert loaded == shape

#------------------------------------------------------------------------------
def test_json_holds_only_the_polygon():
    shape = NuggetShape("box", [(0, 0), (30, 0), (30, 20)])

    assert shape.to_json_dict() == {"polygon": [[0, 0], [30, 0], [30, 20]]}

#------------------------------------------------------------------------------
def test_pixel_contour_is_converted_to_millimeters():
    contour = make_rectangle_contour(500, 800, 300, 200)

    shape = NuggetShape.from_pixel_contour(
        "box", contour, MM_PER_PX, TOLERANCE_MM
    )

    width, height = shape.bounding_size_mm
    assert width == pytest.approx(30, abs=0.01)
    assert height == pytest.approx(20, abs=0.01)
    assert shape.area_mm2 == pytest.approx(600, rel=0.001)

#------------------------------------------------------------------------------
def test_rectangle_is_simplified_to_its_corners():
    contour = make_rectangle_contour(500, 800, 300, 200)

    shape = NuggetShape.from_pixel_contour(
        "box", contour, MM_PER_PX, TOLERANCE_MM
    )

    assert len(shape.polygon_mm) == 4

#------------------------------------------------------------------------------
def test_polygon_is_centered_on_its_centroid():
    contour = make_rectangle_contour(500, 800, 300, 200)

    shape = NuggetShape.from_pixel_contour(
        "box", contour, MM_PER_PX, TOLERANCE_MM
    )

    xs = [x for x, _ in shape.polygon_mm]
    ys = [y for _, y in shape.polygon_mm]
    assert (min(xs), max(xs)) == pytest.approx((-15, 15), abs=0.01)
    assert (min(ys), max(ys)) == pytest.approx((-10, 10), abs=0.01)

#------------------------------------------------------------------------------
def test_polygon_stays_within_the_tolerance_of_the_outline():
    radius_px = 400
    contour = make_circle_contour(2000, 3000, radius_px)

    shape = NuggetShape.from_pixel_contour(
        "circle", contour, MM_PER_PX, TOLERANCE_MM
    )

    # the polygon points lie on the outline, which has a radius of 40 mm
    distances = [np.hypot(x, y) for x, y in shape.polygon_mm]
    assert max(distances) == pytest.approx(40, abs=0.05)

    # a polygon with corners cuts inside the outline by at most the tolerance
    assert shape.area_mm2 <= np.pi * 40 ** 2
    assert shape.area_mm2 >= np.pi * (40 - TOLERANCE_MM) ** 2

#------------------------------------------------------------------------------
def test_simplifying_removes_most_of_the_points():
    contour = make_circle_contour(2000, 3000, 400)

    shape = NuggetShape.from_pixel_contour(
        "circle", contour, MM_PER_PX, TOLERANCE_MM
    )

    assert len(shape.polygon_mm) < len(contour) / 10

#------------------------------------------------------------------------------
def test_points_are_rounded():
    contour = make_circle_contour(2000, 3000, 400)

    shape = NuggetShape.from_pixel_contour(
        "circle", contour, MM_PER_PX, TOLERANCE_MM
    )

    for x, y in shape.polygon_mm:
        assert x == round(x, 2)
        assert y == round(y, 2)

#------------------------------------------------------------------------------
def test_points_run_the_same_way_whichever_way_the_outline_did():
    contour = make_rectangle_contour(500, 800, 300, 200)

    forward = NuggetShape.from_pixel_contour(
        "box", contour, MM_PER_PX, TOLERANCE_MM
    )
    backward = NuggetShape.from_pixel_contour(
        "box", contour[::-1], MM_PER_PX, TOLERANCE_MM
    )

    assert NuggetShape._signed_area(forward.polygon_mm, signed=True) > 0
    assert NuggetShape._signed_area(backward.polygon_mm, signed=True) > 0

