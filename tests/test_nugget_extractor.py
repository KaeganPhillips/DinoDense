###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# test_nugget_extractor.py
#
# Tests for the nugget extractor. Scans are mock images saved as files (see
# mock_scans.py), and shapes and overlays are saved to a temporary folder.
# Showing the overlay opens a window, so it is replaced with a recorder.
###############################################################################

import json
import sys

import cv2
import numpy as np
import pytest

import nugget_extractor
from dime_detector import DIME_DIAMETER_MM, DimeNotFoundError
from mock_scans import draw_mock_scan, make_blank_scan
from nugget_extractor import (
    NuggetExtractor,
    ScanLoadError,
)
from nugget_finder import NuggetNotFoundError
from nugget_shape import NuggetShape

WIDTH = 1000
HEIGHT = 1400
DIME_X = 400
DIME_Y = 300
DIME_RADIUS = 50

#------------------------------------------------------------------------------
@pytest.fixture
def scan_path(tmp_path):
    '''Saves a mock scan named trex.jpg and returns its path.'''
    scan = draw_mock_scan(WIDTH, HEIGHT, DIME_X, DIME_Y, DIME_RADIUS, True)
    path = tmp_path / "scans" / "trex.jpg"
    path.parent.mkdir()
    cv2.imwrite(str(path), scan)

    return path

#------------------------------------------------------------------------------
@pytest.fixture
def extractor(tmp_path):
    '''Makes an extractor that saves into a temporary folder.'''
    return NuggetExtractor(
        shapes_dir=tmp_path / "shapes", overlays_dir=tmp_path / "overlays"
    )

#------------------------------------------------------------------------------
@pytest.fixture
def shown(extractor, monkeypatch):
    '''Replaces showing the overlay and returns the list of shown titles.'''
    titles = []
    monkeypatch.setattr(
        extractor.overlay_renderer, "show",
        lambda overlay, title: titles.append(title)
    )

    return titles

#------------------------------------------------------------------------------
def test_shape_is_named_after_the_scan_file(extractor, scan_path):
    result = extractor.extract(scan_path)

    assert result.shape.name == "trex"

#------------------------------------------------------------------------------
def test_shape_is_scaled_from_the_dime(extractor, scan_path):
    result = extractor.extract(scan_path)

    mm_per_px = DIME_DIAMETER_MM / (2 * DIME_RADIUS)
    assert result.mm_per_px == pytest.approx(mm_per_px, rel=0.02)

    # the mock nugget is an ellipse with half axes of 100 and 150 pixels. A
    # polygon cuts inside a smooth curve, so it is allowed to be a little
    # smaller than the ellipse
    drawn_area_mm2 = np.pi * 100 * 150 * mm_per_px ** 2
    assert result.shape.area_mm2 == pytest.approx(drawn_area_mm2, rel=0.05)

#------------------------------------------------------------------------------
def test_shape_is_centered(extractor, scan_path):
    result = extractor.extract(scan_path)

    points = np.array(result.shape.polygon_mm, dtype=np.float32)
    moments = cv2.moments(points)

    assert moments["m10"] / moments["m00"] == pytest.approx(0, abs=0.05)
    assert moments["m01"] / moments["m00"] == pytest.approx(0, abs=0.05)

#------------------------------------------------------------------------------
def test_extract_returns_an_overlay_and_saves_nothing(
    extractor, scan_path, tmp_path
):
    result = extractor.extract(scan_path)

    assert result.overlay.ndim == 3
    assert result.warnings == []
    assert not (tmp_path / "shapes").exists()
    assert not (tmp_path / "overlays").exists()

#------------------------------------------------------------------------------
def test_run_saves_the_shape_and_the_overlay(
    extractor, scan_path, tmp_path, shown
):
    result = extractor.run(scan_path)

    shape_path = tmp_path / "shapes" / "trex.json"
    saved = NuggetShape.from_json_dict(
        "trex", json.loads(shape_path.read_text(encoding="utf-8"))
    )
    assert saved == result.shape

    overlay = cv2.imread(str(tmp_path / "overlays" / "trex.jpg"))
    assert overlay.shape == result.overlay.shape

#------------------------------------------------------------------------------
def test_saved_json_has_one_point_per_line(
    extractor, scan_path, tmp_path, shown
):
    result = extractor.run(scan_path)

    lines = (tmp_path / "shapes" / "trex.json").read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines[:2] == ["{", '  "polygon": [']
    assert lines[-2:] == ["  ]", "}"]
    assert len(lines) == len(result.shape.polygon_mm) + 4

#------------------------------------------------------------------------------
def test_run_overwrites_the_files_of_an_earlier_run(
    extractor, scan_path, tmp_path, shown
):
    shape_path = tmp_path / "shapes" / "trex.json"
    shape_path.parent.mkdir()
    shape_path.write_text("old", encoding="utf-8")

    extractor.run(scan_path)

    assert "polygon" in shape_path.read_text(encoding="utf-8")

#------------------------------------------------------------------------------
def test_run_shows_the_overlay(extractor, scan_path, shown):
    extractor.run(scan_path)

    assert len(shown) == 1
    assert "trex" in shown[0]

#------------------------------------------------------------------------------
def test_run_can_skip_showing_the_overlay(extractor, scan_path, shown):
    extractor.run(scan_path, show_overlay=False)

    assert shown == []

#------------------------------------------------------------------------------
def test_run_prints_warnings(extractor, tmp_path, shown, capsys):
    scan = draw_mock_scan(WIDTH, HEIGHT, DIME_X, DIME_Y, DIME_RADIUS, False)

    # a nugget cut off by the bottom of the scan
    cv2.ellipse(scan, (DIME_X, HEIGHT - 20), (100, 150), 20, 0, 360,
                (40, 140, 220), -1)
    path = tmp_path / "cut.jpg"
    cv2.imwrite(str(path), scan)

    extractor.run(path)

    assert "Warning: The nugget touches the edge" in capsys.readouterr().out

#------------------------------------------------------------------------------
def test_missing_scan_raises(extractor, tmp_path):
    with pytest.raises(ScanLoadError):
        extractor.extract(tmp_path / "nothing.jpg")

#------------------------------------------------------------------------------
def test_file_that_is_not_an_image_raises(extractor, tmp_path):
    path = tmp_path / "notes.jpg"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ScanLoadError):
        extractor.extract(path)

#------------------------------------------------------------------------------
def test_scan_without_a_dime_raises(extractor, tmp_path):
    path = tmp_path / "nodime.jpg"
    cv2.imwrite(str(path), make_blank_scan(WIDTH, HEIGHT))

    with pytest.raises(DimeNotFoundError):
        extractor.extract(path)

#------------------------------------------------------------------------------
def test_scan_without_a_nugget_raises(extractor, tmp_path):
    path = tmp_path / "nonugget.jpg"
    scan = draw_mock_scan(WIDTH, HEIGHT, DIME_X, DIME_Y, DIME_RADIUS, False)
    cv2.imwrite(str(path), scan)

    with pytest.raises(NuggetNotFoundError):
        extractor.extract(path)

#------------------------------------------------------------------------------
def test_main_extracts_the_scan_named_on_the_command_line(
    scan_path, tmp_path, monkeypatch
):
    monkeypatch.setattr(nugget_extractor, "SHAPES_DIR", tmp_path / "shapes")
    monkeypatch.setattr(
        nugget_extractor, "OVERLAYS_DIR", tmp_path / "overlays"
    )
    monkeypatch.setattr(
        sys, "argv", ["nugget_extractor.py", str(scan_path), "--no-show"]
    )

    nugget_extractor.main()

    assert (tmp_path / "shapes" / "trex.json").is_file()
    assert (tmp_path / "overlays" / "trex.jpg").is_file()

#------------------------------------------------------------------------------
def test_main_reports_an_error_and_exits_with_failure(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(nugget_extractor, "SHAPES_DIR", tmp_path / "shapes")
    monkeypatch.setattr(
        nugget_extractor, "OVERLAYS_DIR", tmp_path / "overlays"
    )
    monkeypatch.setattr(
        sys, "argv", ["nugget_extractor.py", str(tmp_path / "nothing.jpg")]
    )

    with pytest.raises(SystemExit) as exit_info:
        nugget_extractor.main()

    assert exit_info.value.code == 1
    assert "Cannot extract: Scan not found" in capsys.readouterr().out
    assert not (tmp_path / "shapes").exists()

