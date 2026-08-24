import cv2

class MOSSETracker:
    def __init__(self):
        self.tracker = None
        self.box_coords = None
        self.initialized = False

    def create_tracker(self):
        tracker = None
        try:
            # OpenCV'nin standart veya legacy yollarını dene
            if hasattr(cv2, 'TrackerMOSSE_create'):
                tracker = cv2.TrackerMOSSE_create()
            elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
                tracker = cv2.legacy.TrackerMOSSE_create()
            elif hasattr(cv2, 'TrackerKCF_create'): # Yedek olarak KCF dene
                tracker = cv2.TrackerKCF_create()
        except Exception as e:
            print(f"Tracker oluşturulamadı: {e}")
            
        return tracker

    def initialize(self, frame, obj):
        self.reset()
        self.tracker = self.create_tracker()
        
        if self.tracker is None:
            # Eğer sistemde hiçbir tracker desteklenmiyorsa sahte koordinat dönerek çökmesini önle
            x, y, w, h = obj
            self.box_coords = (int(x), int(y), int(w), int(h))
            self.initialized = True
            return self.box_coords

        x, y, w, h = obj
        self.box_coords = (int(x), int(y), int(w), int(h))
        
        try:
            result = self.tracker.init(frame, self.box_coords)
            ok = True if result is None else result
        except Exception:
            ok = False

        if not ok:
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
                return False
            x, y, w, h = box
            self.box_coords = (int(x), int(y), int(w), int(h))
            return True
        except Exception:
            return False

    def get_box(self):
        return self.box_coords

    def reset(self):
        self.tracker = None
        self.box_coords = None
        self.initialized = False

    def draw_tracks(self, frame, old_points=None, new_points=None):
        return frame