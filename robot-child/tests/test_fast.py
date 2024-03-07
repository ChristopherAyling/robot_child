import numpy as np
from robot_child.orb import fast
import cv2

np.random.seed(0)


def test_moments_vs_cv2():
    img = np.random.rand(5, 5)
    m00 = fast.make_moment(img, 0, 0)
    m01 = fast.make_moment(img, 0, 1)
    m10 = fast.make_moment(img, 1, 0)
    cv2_moments = cv2.moments(img)
    atol = 2
    assert np.allclose(m00, cv2_moments["m00"], atol=atol)
    assert np.allclose(m01, cv2_moments["m01"], atol=atol)
    assert np.allclose(m10, cv2_moments["m10"], atol=atol)


def test_make_moment_area():
    img = np.zeros((2, 2))
    m = fast.make_moment(img, 0, 0)
    assert m == 0

    img = np.ones((2, 2))
    m = fast.make_moment(img, 0, 0)
    assert m == 4

    img = np.zeros((2, 2))
    img[0, 0] = 1
    m = fast.make_moment(img, 0, 0)
    assert m == 1


def test_make_centroid():
    img = np.ones((3, 3), dtype=np.float32)
    moments = cv2.moments(img)
    cv2_centroid = (moments["m01"] / moments["m00"], moments["m10"] / moments["m00"])
    cx, cy = fast.make_centroid(img)
    assert (cx, cy) == cv2_centroid
    assert cx == 1
    assert cy == 1

    img = np.zeros((3, 3), dtype=np.float32)
    img[0, 0] = 1
    moments = cv2.moments(img)
    cv2_centroid = (moments["m01"] / moments["m00"], moments["m10"] / moments["m00"])
    cx, cy = fast.make_centroid(img)
    assert (cx, cy) == cv2_centroid
    assert cx == 0
    assert cy == 0
