from contextlib import contextmanager
import time
import cv2


def point_grid(height, width, step):
    margin = 10
    points = []
    for i in range(margin, height - margin, step):
        for j in range(margin, width - margin, step):
            points.append((i, j))
    return points


def preprocess_frame(frame, img_scale):
    frame = cv2.resize(
        frame, (frame.shape[1] // img_scale, frame.shape[0] // img_scale)
    )
    return frame


def preprocess_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


@contextmanager
def timer(name):
    s = time.perf_counter()
    yield
    e = time.perf_counter()
    print(f"{name} Time: {e - s:.4f}s")
