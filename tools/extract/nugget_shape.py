###############################################################################
# Kaegan Phillips
# November 15, 2026
#
# nugget_shape.py
#
# The shape of one kind of dino-nugget: its name and its outline as a polygon
# in millimeters. This is what is saved to data/shapes/ and what the packer
# reads, so it holds nothing about the scan the shape came from.
###############################################################################

from dataclasses import dataclass

import cv2
import numpy as np

#------------------------------------------------------------------------------
# polygon points are rounded to this many decimal places in millimeters
COORDINATE_DECIMALS = 2

#------------------------------------------------------------------------------
@dataclass
class NuggetShape:
    '''
    A dino-nugget shape.

    name: the name of the shape, e.g. "trex"
    polygon_mm: the outline as a list of (x, y) points in millimeters, centered
                on the centroid of the shape. The y axis points down, as in
                the scan, and the points run clockwise as seen on screen.
    '''
    name: str
    polygon_mm: list

    #--------------------------------------------------------------------------
    @property
    def area_mm2(self):
        '''The area inside the polygon in square millimeters.'''
        return self._signed_area(self.polygon_mm)

    #--------------------------------------------------------------------------
    @property
    def bounding_size_mm(self):
        '''The width and height of the polygon as a tuple in millimeters.'''
        xs = [x for x, _ in self.polygon_mm]
        ys = [y for _, y in self.polygon_mm]

        return max(xs) - min(xs), max(ys) - min(ys)

    #--------------------------------------------------------------------------
    def to_json_dict(self):
        '''
        Method to get the shape as a dictionary for saving as JSON. The name is
        not included, since it is the name of the file.

        return: a dictionary with the polygon as a list of [x, y] lists
        '''
        return {"polygon": [[x, y] for x, y in self.polygon_mm]}

    #--------------------------------------------------------------------------
    @classmethod
    def from_json_dict(cls, name, json_dict):
        '''
        Method to make a shape from a dictionary loaded from JSON.

        param: name: the name of the shape, e.g. the name of the file
        param: json_dict: the dictionary made by to_json_dict
        return: a NuggetShape
        '''
        polygon_mm = [(x, y) for x, y in json_dict["polygon"]]

        return cls(name=name, polygon_mm=polygon_mm)

    #--------------------------------------------------------------------------
    @classmethod
    def from_pixel_contour(cls, name, contour_px, mm_per_px, tolerance_mm):
        '''
        Method to make a shape from an outline found in a scan. The outline is
        converted to millimeters, simplified to a polygon, and centered on its
        centroid.

        param: name: the name of the shape, e.g. "trex"
        param: contour_px: the outline as an array of (x, y) points in pixels
        param: mm_per_px: the millimeters that one pixel of the scan covers
        param: tolerance_mm: how far in millimeters the polygon may stray from
               the outline. Larger values give fewer points.
        return: a NuggetShape
        '''
        polygon_px = cls.simplify_contour(contour_px, mm_per_px, tolerance_mm)

        return cls.from_pixel_polygon(name, polygon_px, mm_per_px)

    #--------------------------------------------------------------------------
    @classmethod
    def from_pixel_polygon(cls, name, polygon_px, mm_per_px):
        '''
        Method to make a shape from a polygon in scan pixels. The polygon is
        converted to millimeters and centered on its centroid.

        param: name: the name of the shape, e.g. "trex"
        param: polygon_px: the polygon as an array of (x, y) points in pixels
        param: mm_per_px: the millimeters that one pixel of the scan covers
        return: a NuggetShape
        '''
        polygon = (np.asarray(polygon_px) * mm_per_px).astype(np.float32)

        moments = cv2.moments(polygon)
        centroid = np.array([
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ])
        centered = np.round(polygon - centroid, COORDINATE_DECIMALS)

        polygon_mm = [(float(x), float(y)) for x, y in centered]

        # make every shape run the same way round, whatever way the scan did
        if cls._signed_area(polygon_mm, signed=True) < 0:
            polygon_mm.reverse()

        return cls(name=name, polygon_mm=polygon_mm)

    #--------------------------------------------------------------------------
    @staticmethod
    def simplify_contour(contour_px, mm_per_px, tolerance_mm):
        '''
        Method to simplify an outline found in a scan to a polygon with far
        fewer points, using the Douglas-Peucker algorithm. The polygon stays in
        scan pixels, so it can be drawn on the scan.

        param: contour_px: the outline as an array of (x, y) points in pixels
        param: mm_per_px: the millimeters that one pixel of the scan covers
        param: tolerance_mm: how far in millimeters the polygon may stray from
               the outline. Larger values give fewer points.
        return: the polygon as an array of (x, y) points in pixels
        '''
        contour = np.asarray(contour_px, dtype=np.float32)

        return cv2.approxPolyDP(
            contour.reshape(-1, 1, 2), tolerance_mm / mm_per_px, True
        ).reshape(-1, 2)

    #--------------------------------------------------------------------------
    @staticmethod
    def _signed_area(points, signed=False):
        '''
        Method to get the area inside a polygon with the shoelace formula.

        param: points: a list of (x, y) points
        param: signed: True to keep the sign, which is positive for points
               that run clockwise as seen with the y axis pointing down
        return: the area of the polygon
        '''
        total = 0.0

        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
            total += x1 * y2 - x2 * y1

        area = total / 2

        return area if signed else abs(area)

