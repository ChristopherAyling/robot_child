import numpy as np
from numba import njit
from . import draw

"""
Implements the FAST algorithm and visualisation
"""

IntPairs = list[tuple[int, int]]

test_points_full = [
        (0, -3),
        (1, -3),

        (2, -2),

        (3, -1),
        (3, 0),
        (3, 1),

        (2, 2),

        (1, 3),
        (0, 3),
        (-1, 3),

        (-2, 2),

        (-3, 1),
        (-3, 0),
        (-3, -1),

        (-2, -2),

        (-1, -3)
    ]
assert len(test_points_full) == len(set(test_points_full))

test_points_fast = [test_points_full[i-1] for i in [1, 9, 5, 13]]

@njit
def make_moment(img, p, q):
    # TODO circular patch
    moment = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            moment += (i**p)*(j**q)*img[i, j]
    return moment

@njit
def make_centroid(patch: np.ndarray) -> tuple[float, float]:
    m00 = make_moment(patch, 0, 0)
    m01 = make_moment(patch, 0, 1)
    m10 = make_moment(patch, 1, 0)
    return (m01/m00, m10/m00)

@njit
def make_corner_points(img: np.ndarray, grid: IntPairs, test_points: IntPairs, threshold: int) -> IntPairs:
    n = 3
    corner_points = []
    for i, j in grid:
        center_value = img[i, j]
        top = center_value + threshold
        bottom = center_value - threshold
        count = 0
        for (px, py) in test_points:
            x = i + px
            y = j + py
            value = img[x, y]
            if value > top or value < bottom:
                count += 1
                if count >= n:
                    corner_points.append((i, j))
                    break
    return corner_points

@njit
def make_orientations(img: np.ndarray, corner_points: IntPairs, margin: int) -> list[float]:
    """
    img: grayscale image between 0 and 1
    corner_points: list of (i, j) tuples
    margin: size of the patch/2 to consider for orientation

    returns: orientations in radians
    """
    orientations = []
    for (i, j) in corner_points:
        patch = img[i-margin:i+margin, j-margin:j+margin]
        cx, cy = make_centroid(patch)
        orientation = np.arctan2(cy-margin, cx-margin)
        orientations.append(orientation)
    return orientations

def harris_measure(img, corner_points):
    # TODO implement
    return [1 for _ in corner_points]

def nms(corner_points, cornerness):
    # TODO implement
    return corner_points


def fast_detection(img, grid, test_points, threshold, margin):
    corner_points = make_corner_points(img, grid=grid, test_points=test_points, threshold=threshold)
    orientations = make_orientations(img/255, corner_points, margin=margin)
    cornerness = harris_measure(img, corner_points)
    _nms_corner_points = nms(corner_points, cornerness)
    return corner_points, cornerness, orientations 

# visualisation

def draw_test_points(img, i, j):
    for point in test_points_full:
        x = i + point[0]
        y = j + point[1]
        draw.set_pixel_rgb(img, x, y, 0, 255, 0)

    for point in test_points_fast:
        x = i + point[0]
        y = j + point[1]
        draw.set_pixel_rgb(img, x, y, 255, 0, 0)

def draw_fast(img, corner_points):
    for (i, j) in corner_points:
        draw_test_points(img, i, j)
