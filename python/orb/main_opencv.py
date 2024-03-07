import video
import utils
from utils import timer
import cv2
import numpy as np
import time

def main():
    img_scale = 4
    orb = cv2.ORB_create()

    for i_frame, frame in enumerate(video.video_iterator(0)):
        with timer("Total"):
            frame = utils.preprocess_frame(frame, img_scale=img_scale)
            gray = utils.preprocess_gray(frame)
            print(gray.shape)
            kp = orb.detect(frame, None)
            kp, des = orb.compute(frame, kp)
            frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
            cv2.imshow("Frame", cv2.resize(frame, dsize=None, fx=img_scale, fy=img_scale))
            if cv2.waitKey(1) == ord("q"):
                break

if __name__ == "__main__":
    main()

