from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

class DeepSortWrapper:
    def __init__(self, yolo_detector):
        self.model = yolo_detector.model
        self.allowed_classes = yolo_detector.allowed_classes
        self.min_width = yolo_detector.min_width
        self.min_height = yolo_detector.min_height
        self.track_class_map = {}
        
        self.deepsort = DeepSort(max_age=60, n_init=3, nn_budget=100)
        
        mask_raw = cv2.imread("trafikvideo.jpg", cv2.IMREAD_GRAYSCALE)
        if mask_raw is not None:
            _, self.mask_bin = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)
        else:
            self.mask_bin = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        conf_thresh = 0.30 if mean_brightness < 90 else 0.40

        results = self.model.predict(
            source=frame,
            conf=conf_thresh,
            classes=self.allowed_classes,
            imgsz=640,
            max_det=100,
            agnostic_nms=True,
            verbose=False
        )

        detections = []
        frame_h, frame_w = frame.shape[:2]
        
        if self.mask_bin is not None:
            current_mask = cv2.resize(self.mask_bin, (frame_w, frame_h))
        else:
            current_mask = np.full((frame_h, frame_w), 255, dtype=np.uint8)

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                x1 = int(max(0, x1))
                y1 = int(max(0, y1))
                x2 = int(min(frame_w - 1, x2))
                y2 = int(min(frame_h - 1, y2))

                bw = x2 - x1
                bh = y2 - y1

                if bw < self.min_width or bh < self.min_height:
                    continue
                if bw > frame_w * 0.95 or bh > frame_h * 0.95:
                    continue
                    
                cx = int(x1 + (bw / 2))
                cy = int(y1 + (bh / 2))
                
                if 0 <= cx < frame_w and 0 <= cy < frame_h:
                    if current_mask[cy, cx] == 0:
                        continue
                else:
                    continue
                    
                aspect_ratio = bw / float(max(1, bh))
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    continue

                detections.append(([x1, y1, bw, bh], conf, cls_id))

        tracks = self.deepsort.update_tracks(detections, frame=frame)
        
        tracked_objects = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            if track.time_since_update > 1:
                continue
            
            track_id = int(track.track_id)
            cls_id = getattr(track, 'det_class', self.track_class_map.get(track_id, 0))
            self.track_class_map[track_id] = cls_id
            
            conf = getattr(track, 'det_conf', None)
            if conf is None:
                conf = 0.85 
            
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            bw = x2 - x1
            bh = y2 - y1
            
            cx = int(x1 + bw/2)
            cy = int(y1 + bh/2)
            
            if 0 <= cx < frame_w and 0 <= cy < frame_h:
                if current_mask[cy, cx] == 0:
                    continue
            else:
                continue
            
            tracked_objects[track_id] = (x1, y1, bw, bh, cls_id, conf)

        return tracked_objects