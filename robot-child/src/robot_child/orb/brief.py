from numba import njit
import numpy as np
from . import draw

def make_brief_point_pairs(patch_size, n):
    point_pairs = []
    bottom = -patch_size // 2
    top = patch_size // 2
    for _ in range(n):
        x1 = np.random.randint(bottom, top)
        y1 = np.random.randint(bottom, top)
        x2 = np.random.randint(bottom, top)
        y2 = np.random.randint(bottom, top)
        point_pairs.append(((x1, y1), (x2, y2)))
    return point_pairs


@njit
def hamming_distance(a, b):
    assert len(a) == len(b)
    return sum([1 for x, y in zip(a, b) if x != y])

@njit
def make_binary_descriptors(img, corner_points, threshold, point_pairs):
    descriptors = []
    for (i, j) in corner_points:
        descriptor = []
        for (p1, p2) in point_pairs:
            x1, y1 = p1
            x2, y2 = p2
            if img[i+x1, j+y1] > img[i+x2, j+y2]:
                descriptor.append(1)
            else:
                descriptor.append(0)
        descriptors.append(descriptor)
    return descriptors


def brief_descriptors(img, corner_points, threshold, point_pairs):
    descriptors = make_binary_descriptors(img, corner_points, threshold, point_pairs)
    return descriptors

# @njit
def hamming_distance(a, b) -> int:
    assert len(a) == len(b)
    return sum([1 for x, y in zip(a, b) if x != y])

# @njit
def make_distance_matrix(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    distances: list[list[int]] = []
    for da in a:
        drow: list[int] = []
        for db in b:
            drow.append(hamming_distance(da, db))
        distances.append(drow)
    return distances

def filter_matches(distances, n, corner_points, old_corner_points):
    best = np.where(distances < np.sort(distances.flatten())[n])

    errors = []
    for i_old, i in zip(*best):
        errors.append(np.linalg.norm(np.array(corner_points[i]) - np.array(old_corner_points[i_old])))
    error_threshold = np.mean(errors)

    matches = []
    for error, i_old, i in zip(errors, *best):
        if error < error_threshold:
            matches.append((old_corner_points[i_old], corner_points[i]))
    return matches

# visualisation

def draw_point_pairs(img, point_pairs):
    for i, ((x1, y1), (x2, y2)) in enumerate(point_pairs):
        draw.set_pixel_rgb(img, x1, y1, 0, 255-i, 0)
        draw.set_pixel_rgb(img, x2, y2, 0, 255-i, 0)

def draw_matches(img, matches):
    for (i_old, i) in matches:
        # draw.line(img, old["corner_points"][i_old], corner_points[i], (0, 255, 0))
        import cv2
        # cv2.line(img, i_old, i, (0, 255, 0))
        draw.line(img, i_old, i, 0, 0, 255)
        draw.set_pixel_rgb(img, i_old[0], i_old[1], 255, 0, 0)
        draw.set_pixel_rgb(img, i[0], i[1], 255, 0, 0)
        # cv2.circle(img, i_old, 5, (0, 0, 255))
        # cv2.circle(img, i, 5, (0, 0, 255))