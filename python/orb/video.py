import cv2

def video_iterator(path, step=1):
    cap = cv2.VideoCapture(path)

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            yield frame
        i += 1
    
    cap.release()