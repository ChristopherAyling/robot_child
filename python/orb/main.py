import cv2
import numpy as np
import numpy
import time
from numba import njit

"""
Implementing ORB feature detection and descriptors

Python dev, C++ prod.

(resizing, smoothing, fix distortion etc)
1. [x] FAST corner detection
2. [ ] Harris measures of corner-ness
3. [ ] Filter with non-maximum suppression or top N
4. [ ] Corner orientation  
(might want to do with a pyramid for multi-scale features)

Need to develop quality and performance benchmarks for comparing OpenCV, my Python and My C++.
"""

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
def is_corner(img, i, j, threshold, test_points, fast=True) -> bool:
    n = 3 if fast else 12
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
                return True
    return False


@njit
def make_moment(img, p, q):
    # TODO circular patch
    moment = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            moment += (i**p)*(j**q)*img[i, j]
    return moment

def make_orientation(img):
    return np.arctan2(
        make_moment(img, 1, 0),
        make_moment(img, 0, 1),
    )

BLUE = 0
GREEN = 1
RED = 2

def set_pixel_rgb(img, i, j, r, g, b):
    img[i, j, RED] = r
    img[i, j, GREEN] = g
    img[i, j, BLUE] = b

def draw_test_points(img, i, j):
    for point in test_points_full:
        x = i + point[0]
        y = j + point[1]
        set_pixel_rgb(img, x, y, 0, 255, 0)

    for point in test_points_fast:
        x = i + point[0]
        y = j + point[1]
        set_pixel_rgb(img, x, y, 255, 0, 0)

def point_grid(height, width, step):
    margin = 10
    points = []
    for i in range(margin, height-margin, step):
        for j in range(margin, width-margin, step):
            points.append((i, j))
    return points


def make_patch_points(patch_size, n):
    # TODO change to normal distribution and pair random points as opposed to pairing each one with center
    points = []
    bottom = -patch_size // 2
    top = patch_size // 2
    for _ in range(n):
        x = numpy.random.randint(bottom, top)
        y = numpy.random.randint(bottom, top) 
        points.append((x, y))
    return points

def draw_patch_points(frame, i, j, patch_points):
    for (x, y) in patch_points:
        set_pixel_rgb(frame, x+i, y+j, 255, 0, 0)

def make_binary_descriptor(img, i, j, threshold, patch_points):
    descriptor = []
    center_value = img[i, j]
    for (x, y) in patch_points:
        value = img[i+x, j+y]
        if value > center_value + threshold:
            descriptor.append(1)
        else:
            descriptor.append(0)
    return descriptor

def draw_binary_descriptor(img, i, j, descriptor, patch_points):
    for ((x, y), indicator) in zip(patch_points, descriptor):
        if indicator:
            set_pixel_rgb(img, i+x, j+y, 255, 0, 0)
        else:
            set_pixel_rgb(img, i+x, j+y, 0, 0, 255)

def hamming_distance(a, b):
    assert len(a) == len(b)
    return sum(1 for x, y in zip(a, b) if x != y)

@njit
def make_centroid(patch: np.ndarray) -> tuple[float, float]:
    m00 = make_moment(patch, 0, 0)
    m01 = make_moment(patch, 0, 1)
    m10 = make_moment(patch, 1, 0)
    return (m01/m00, m10/m00)

IntPairs = list[tuple[int, int]]

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

def fast(img, grid, test_points, threshold):
    corner_points = make_corner_points(img, grid=grid, test_points=test_points, threshold=threshold)
    orientations = make_orientations(img, corner_points, margin=10)
    cornerness = harris_measure(img, corner_points)
    nms_corner_points = nms(corner_points, cornerness)
    return corner_points, cornerness, orientations, nms_corner_points


def main():
    cap = cv2.VideoCapture("kitchen.mov")

    img_scale = 3
    corner_threshold = 10
    points = point_grid(180, 320, step=6)
    patch_points = make_patch_points(20, 50)
    margin = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.resize(frame, (frame.shape[1]//img_scale, frame.shape[0]//img_scale))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)


        # Oriented FAST
        s = time.perf_counter()
        _corner_points, _cornerness, corner_points = fast(gray, points, test_points_fast, corner_threshold)
        e = time.perf_counter()
        print(f"FAST Time: {e - s:.4f}s")
        for (i, j) in corner_points:
            draw_test_points(frame, i, j)

        # Corner Orientations
        s = time.perf_counter()
        orientations = []
        for (i, j) in corner_points:
            patch = gray[i-margin:i+margin, j-margin:j+margin] / 255
            cx, cy = make_centroid(patch)
            orientation = np.rad2deg(np.arctan2(cy-margin, cx-margin))
            orientations.append(orientation)
        e = time.perf_counter()
        print(f"Orientation Time: {e - s:.4f}s")

        # BREIF
        # s = time.perf_counter()
        # descriptors = []
        # for (i, j) in corner_points:
        #     descriptor = make_binary_descriptor(gray, i, j, 10, patch_points)
        #     descriptors.append(descriptor) 
        # for (i, j), descriptor in zip(corner_points, descriptors):
        #     draw_binary_descriptor(frame, i, j, descriptor, patch_points)
        # e = time.perf_counter()
        # print(f"BRIEF Time: {e - s:.4f}s")




        cv2.imshow("Frame", cv2.resize(frame, dsize=None, fx=4, fy=4))
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()