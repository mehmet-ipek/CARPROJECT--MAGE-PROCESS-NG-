import cv2
import numpy as np

class OpticalFlowTracker:
    def __init__(self):
        self.previous_gray = None
        self.previous_points = None
        self.box_coords = None
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

    def initialize(self, frame, obj):
        self.box_coords = obj
        x, y, w, h = obj
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255

        points = cv2.goodFeaturesToTrack(
            gray, mask=mask, maxCorners=100, qualityLevel=0.05, minDistance=5
        )

        if points is None:
            return None

        self.previous_gray = gray
        self.previous_points = points
        return points

    def track(self, frame):
        if self.previous_gray is None or self.previous_points is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.previous_points, None, **self.lk_params
        )

        if new_points is None:
            return None

        good_new = new_points[status == 1]
        good_old = self.previous_points[status == 1]

        if len(good_new) == 0:
            return None

        points_x = good_new[:, 0]
        points_y = good_new[:, 1]

        min_x = int(np.min(points_x))
        max_x = int(np.max(points_x))
        min_y = int(np.min(points_y))
        max_y = int(np.max(points_y))

        new_w = max_x - min_x
        new_h = max_y - min_y

        if new_w > 20 and new_h > 20:
            self.box_coords = (min_x, min_y, new_w, new_h)

        self.previous_gray = gray
        self.previous_points = good_new.reshape(-1, 1, 2)

        return (good_old, good_new)

    def get_box(self):
        return self.box_coords

    def reset(self):
        self.previous_gray = None
        self.previous_points = None
        self.box_coords = None

    def draw_tracks(self, frame, old_points, new_points):
        if old_points is None or new_points is None:
            return frame

        for old, new in zip(old_points, new_points):
            a, b = old.ravel()
            c, d = new.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(frame, (int(c), int(d)), 4, (0, 0, 255), -1)

        return frame