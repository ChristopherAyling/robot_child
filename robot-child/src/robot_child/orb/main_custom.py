from . import fast
from . import brief
from . import video
from . import utils
from .utils import timer
import cv2
import numpy as np


def main():
    # height = 180
    # width = 320
    height = 270
    width = 480
    img_scale = 4
    grid = utils.point_grid(height, width, step=4)
    corner_threshold = 10
    margin = 10
    point_pairs = brief.make_brief_point_pairs(margin * 2, n=64)
    brief_threshold = 10
    n_best_matches = 50

    old = None

    for i_frame, frame in enumerate(video.video_iterator(0)):
        with timer("Total"):
            frame = utils.preprocess_frame(frame, img_scale=img_scale)
            gray = utils.preprocess_gray(frame)
            assert gray.shape == (
                height,
                width,
            ), f"Expected shape {(height, width)}, got {gray.shape}"

            with timer("-- FAST"):
                corner_points, cornerness, orientations = fast.fast_detection(
                    gray,
                    grid=grid,
                    test_points=fast.test_points_fast,
                    threshold=corner_threshold,
                    margin=margin,
                )
                print(len(corner_points))
            # fast.draw_fast(frame, corner_points)

            with timer("-- BREIF"):
                descriptors = brief.brief_descriptors(
                    gray, corner_points, brief_threshold, point_pairs
                )
            # draw_brief(frame, corner_points)

            if old is not None:
                with timer("-- Matching"):
                    distances = np.array(
                        brief.make_distance_matrix(old["descriptors"], descriptors)
                    ).astype(np.uint8)
                    matches = brief.filter_matches(
                        distances, n_best_matches, corner_points, old["corner_points"]
                    )
                    print(len(matches))
                    brief.draw_matches(frame, matches)

                # cv2.imshow("Distances", distances)
                # draw_matches()

            old = {"corner_points": corner_points, "descriptors": descriptors}

        cv2.imshow("Frame", cv2.resize(frame, dsize=None, fx=img_scale, fy=img_scale))
        if cv2.waitKey(1) == ord("q"):
            break
        # if i_frame > 5:
        #     time.sleep(10)


if __name__ == "__main__":
    main()
