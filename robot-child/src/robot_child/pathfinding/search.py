import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_map(path, dilation=50):
    map = plt.imread(path)
    map = map[..., 0]
    map[map>0.5] = 1
    map[map<=0.5] = 0
    map = 1-map.astype(np.uint8)
    occ = cv2.dilate(map, np.ones((dilation, dilation), dtype=np.uint8))
    return map, occ

def adjacent_points(px, py, map):
    width, height = map.shape
    assert map[px, py] == 0
    
    options = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if not (dx == 0 and dy == 0):
                if px+dx >= 0 and px+dx<width and py+dy >= 0 and py+dy < height:
                    if map[px+dx, py+dy] == 0:
                        options.append((px+dx, py+dy))
    assert len(options) <= 8, f"too many options {len(options)} returned for {px=}, {py=}, {map.shape=}"
    assert len(options) >= 1, f"no options returned for {px=}, {py=}, {map.shape=}"
    return options
                
def bfs(start, end, map):
    assert map[start[0], start[1]] == 0, f"Starting point {start} must be unoccupied"
    assert map[end[0], end[1]] == 0, f"Ending point {end} must be unoccupied"
    discovered = set()
    # [ (node, [path]), ... ]
    stack = [(start, [start])]    
    while stack:
        node, path = stack.pop(0)
        if not node in discovered:
            if node == end:
                return path
            discovered.add(node)
            for neighbor in adjacent_points(node[0], node[1], map):
                stack.append((neighbor, path + [neighbor]))
    raise ValueError("no valid path found!")