BLUE = 0
GREEN = 1
RED = 2

def set_pixel_rgb(img, i, j, r, g, b):
    img[i, j, RED] = r
    img[i, j, GREEN] = g
    img[i, j, BLUE] = b

def line(img, p1, p2, r, g, b):
    (x1, y1), (x2, y2) = p1, p2
    dx = x2 - x1
    dy = y2 - y1
    for x in range(x1, x2):
        y = y1 + dy * (x - x1) // dx
        set_pixel_rgb(img, x, y, r, g, b)