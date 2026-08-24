import cv2
import numpy as np

class TrafficHeatmap:
    def __init__(self, width=1920, height=1080, decay_factor=0.985):
        self.width = width
        self.height = height
        self.decay_factor = decay_factor
        self.heatmap_accumulator = np.zeros((height, width), dtype=np.float32)

    def update_from_mot(self, tracked_bboxes):
        self.heatmap_accumulator *= self.decay_factor
        for t_id, box in tracked_bboxes.items():
            x1, y1, bw, bh = box
            cx = int(x1 + bw / 2)
            cy = int(y1 + bh / 2)
            self._add_heat_point(cx, cy, radius=45, intensity=60.0)

    def update_from_centroid(self, objects_dict, ct_boxes):
        self.heatmap_accumulator *= self.decay_factor
        for obj_id, centroid in objects_dict.items():
            cx, cy = centroid
            self._add_heat_point(cx, cy, radius=45, intensity=60.0)

    def update_from_single_tracker(self, box):
        self.heatmap_accumulator *= self.decay_factor
        if box is not None:
            bx, by, bw, bh = box
            cx = int(bx + bw / 2)
            cy = int(by + bh / 2)
            self._add_heat_point(cx, cy, radius=50, intensity=70.0)

    def _add_heat_point(self, cx, cy, radius, intensity):
        if 0 <= cx < self.width and 0 <= cy < self.height:
            cv2.circle(self.heatmap_accumulator, (cx, cy), radius=radius, color=intensity, thickness=-1)

    def get_colored_heatmap(self, frame):
        h, w = frame.shape[:2]

        normalized = cv2.normalize(self.heatmap_accumulator, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_gray = normalized.astype(np.uint8)

        if heatmap_gray.shape[:2] != (h, w):
            heatmap_gray = cv2.resize(heatmap_gray, (w, h))

        color_heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        mask = heatmap_gray > 8

        blended_frame = frame.copy()
        if np.any(mask):
            roi_frame = frame[mask]
            roi_heat = color_heatmap[mask]
            blended = cv2.addWeighted(roi_frame, 0.5, roi_heat, 0.5, 0)
            blended_frame[mask] = blended

        return blended_frame

    def reset(self):
        self.heatmap_accumulator = np.zeros((self.height, self.width), dtype=np.float32)