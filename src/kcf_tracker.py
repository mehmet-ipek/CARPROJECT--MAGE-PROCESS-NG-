import cv2

class KCFTracker:
    def __init__(self):
        self.tracker = None
        self.box_coords = None
        data = None
        self.initialized = False

    def create_tracker(self):
        try:
            return cv2.TrackerKCF_create()
        except AttributeError:
            return cv2.legacy.TrackerKCF_create()

    def initialize(self, frame, obj):
        self.reset()
        self.tracker = self.create_tracker()
        if obj is None or len(obj) < 4:
            return None
            
        x, y, w, h = obj
        expand = 10
        x = max(0, x - expand)
        y = max(0, y - expand)
        w = w + expand * 2
        h = h + expand * 2

        self.box_coords = (int(x), int(y), int(w), int(h))
        try:
            result = self.tracker.init(frame, self.box_coords)
            ok = True if result is None else result
            if not ok:
                self.reset()
                return None
        except Exception:
            self.reset()
            return None

        self.initialized = True
        return self.box_coords

    def track(self, frame):
        if not self.initialized or self.tracker is None:
            return False

        try:
            success, box = self.tracker.update(frame)
            if not success:
                self.reset()
                return False

            x, y, w, h = box
            self.box_coords = (int(x), int(y), int(w), int(h))
            return True
        except Exception:
            self.reset()
            return False

    def get_box(self):
        return self.box_coords

    def reset(self):
        self.tracker = None
        self.box_coords = None
        self.initialized = False

    def draw_tracks(self, frame, old_points=None, new_points=None):
        return frame