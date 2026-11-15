###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# test_overlay_renderer.py
#
# Tests for the overlay renderer. Scans are drawn as mock images (see
# mock_scans.py). Showing an overlay opens a window, so the window functions
# of OpenCV are replaced with a fake window.
###############################################################################

import cv2
import numpy as np
import pytest

from dime_detector import DimeDetection
from mock_scans import draw_mock_scan
from overlay_renderer import (
    DIME_COLOR,
    OVERLAY_MAX_SIZE_PX,
    POLYGON_COLOR,
    OverlayRenderer,
)

WINDOW_TITLE = "box"

WIDTH = 4000
HEIGHT = 5000
DIME_X = 1600
DIME_Y = 1200
DIME_RADIUS = 160

# a square standing in for the nugget polygon, below the dime
POLYGON = [(1400, 2000), (2200, 2000), (2200, 3200), (1400, 3200)]

#------------------------------------------------------------------------------
def make_scan_and_dime():
    '''
    Method to make a mock scan and the DimeDetection that matches its dime.

    return: a tuple of the scan and the DimeDetection
    '''
    scan = draw_mock_scan(WIDTH, HEIGHT, DIME_X, DIME_Y, DIME_RADIUS, False)

    return scan, DimeDetection(DIME_X, DIME_Y, DIME_RADIUS)

#------------------------------------------------------------------------------
def has_color(image, color):
    '''
    Method to check if an image has any pixel of exactly one color.

    param: image: a BGR image array
    param: color: a BGR color tuple
    return: True if any pixel has the color
    '''
    return bool(np.any(np.all(image == color, axis=2)))

#------------------------------------------------------------------------------
def test_overlay_shows_the_dime_circle_and_the_polygon():
    scan, dime_detection = make_scan_and_dime()

    overlay = OverlayRenderer().render(scan, dime_detection, POLYGON)

    assert has_color(overlay, DIME_COLOR)
    assert has_color(overlay, POLYGON_COLOR)

#------------------------------------------------------------------------------
def test_overlay_is_cropped_and_shrunk():
    scan, dime_detection = make_scan_and_dime()

    overlay = OverlayRenderer().render(scan, dime_detection, POLYGON)

    assert max(overlay.shape[:2]) == OVERLAY_MAX_SIZE_PX
    assert overlay.shape[0] > overlay.shape[1]  # the crop is taller than wide

#------------------------------------------------------------------------------
def test_overlay_keeps_the_aspect_ratio_of_the_crop():
    scan, dime_detection = make_scan_and_dime()

    overlay = OverlayRenderer().render(scan, dime_detection, POLYGON)

    # the dime and polygon span x 1400-2200 and y 1040-3200, plus a margin of
    # 10% of the larger side (2160) on every side
    expected_ratio = (800 + 2 * 216) / (2160 + 2 * 216)
    assert overlay.shape[1] / overlay.shape[0] == pytest.approx(
        expected_ratio, rel=0.01
    )

#------------------------------------------------------------------------------
def test_small_overlay_is_not_enlarged():
    scan = draw_mock_scan(500, 600, 100, 100, 30, False)
    dime_detection = DimeDetection(100, 100, 30)
    polygon = [(80, 200), (180, 200), (180, 300), (80, 300)]

    overlay = OverlayRenderer().render(scan, dime_detection, polygon)

    assert max(overlay.shape[:2]) < OVERLAY_MAX_SIZE_PX
    assert max(overlay.shape[:2]) < 500

#------------------------------------------------------------------------------
def test_render_does_not_change_the_scan():
    scan, dime_detection = make_scan_and_dime()
    original = scan.copy()

    OverlayRenderer().render(scan, dime_detection, POLYGON)

    assert np.array_equal(scan, original)

#------------------------------------------------------------------------------
def test_crop_stays_inside_a_scan_when_the_dime_is_near_the_edge():
    scan = draw_mock_scan(1000, 1400, 40, 40, 30, False)
    dime_detection = DimeDetection(40, 40, 30)
    polygon = [(200, 300), (500, 300), (500, 700), (200, 700)]

    overlay = OverlayRenderer().render(scan, dime_detection, polygon)

    assert has_color(overlay, DIME_COLOR)
    assert has_color(overlay, POLYGON_COLOR)

#------------------------------------------------------------------------------
def test_save_writes_a_readable_image_and_makes_the_folder(tmp_path):
    scan, dime_detection = make_scan_and_dime()
    renderer = OverlayRenderer()
    overlay = renderer.render(scan, dime_detection, POLYGON)
    path = tmp_path / "overlays" / "box.jpg"

    renderer.save(overlay, path)

    loaded = cv2.imread(str(path))
    assert loaded.shape == overlay.shape

#------------------------------------------------------------------------------
def test_save_overwrites_an_existing_overlay(tmp_path):
    renderer = OverlayRenderer()
    path = tmp_path / "box.jpg"
    first = np.zeros((50, 60, 3), dtype=np.uint8)
    second = np.zeros((80, 90, 3), dtype=np.uint8)

    renderer.save(first, path)
    renderer.save(second, path)

    assert cv2.imread(str(path)).shape == second.shape

#------------------------------------------------------------------------------
def test_save_raises_when_the_image_cannot_be_written(tmp_path):
    overlay = np.zeros((50, 60, 3), dtype=np.uint8)

    with pytest.raises(Exception):
        OverlayRenderer().save(overlay, tmp_path / "box.notanimage")

#------------------------------------------------------------------------------
class FakeWindow:
    '''
    A stand-in for the OpenCV window functions. The window stays visible for
    the given number of checks, then counts as closed by hand. Any key press
    in key_presses is returned by waitKey in turn.
    '''

    def __init__(self, visible_checks, key_presses=()):
        self.visible_checks = visible_checks
        self.key_presses = list(key_presses)
        self.is_destroyed = False

    def get_window_property(self, title, prop):
        if self.visible_checks == 0:
            return -1  # closed by hand

        self.visible_checks -= 1
        return 1

    def wait_key(self, delay):
        return self.key_presses.pop(0) if self.key_presses else -1

    def destroy_window(self, title):
        # as OpenCV does when the window is already gone
        if self.visible_checks == 0:
            raise cv2.error("NULL window")

        self.is_destroyed = True

    def destroy_all_windows(self):
        self.is_destroyed = True

#------------------------------------------------------------------------------
def install_fake_window(monkeypatch, fake_window):
    '''
    Method to replace the OpenCV window functions with a fake window.

    param: monkeypatch: the pytest monkeypatch fixture
    param: fake_window: the FakeWindow to replace the functions with
    '''
    monkeypatch.setattr(cv2, "namedWindow", lambda title, flags: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda title, w, h: None)
    monkeypatch.setattr(cv2, "imshow", lambda title, image: None)
    monkeypatch.setattr(
        cv2, "getWindowProperty", fake_window.get_window_property
    )
    monkeypatch.setattr(cv2, "waitKey", fake_window.wait_key)
    monkeypatch.setattr(cv2, "destroyWindow", fake_window.destroy_window)
    monkeypatch.setattr(
        cv2, "destroyAllWindows", fake_window.destroy_all_windows
    )

#------------------------------------------------------------------------------
def test_show_closes_when_a_key_is_pressed(monkeypatch):
    fake_window = FakeWindow(visible_checks=100, key_presses=[-1, -1, 32])
    install_fake_window(monkeypatch, fake_window)
    overlay = np.zeros((50, 60, 3), dtype=np.uint8)

    OverlayRenderer().show(overlay, WINDOW_TITLE)

    assert fake_window.is_destroyed

#------------------------------------------------------------------------------
def test_show_closes_when_the_window_is_closed_by_hand(monkeypatch):
    fake_window = FakeWindow(visible_checks=3)
    install_fake_window(monkeypatch, fake_window)
    overlay = np.zeros((50, 60, 3), dtype=np.uint8)

    OverlayRenderer().show(overlay, WINDOW_TITLE)

    assert fake_window.is_destroyed

