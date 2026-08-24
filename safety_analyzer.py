import math
import cv2
import numpy as np
import os
from collections import defaultdict, deque
from ultralytics import YOLO

class SafetyAnalyzer:
    def __init__(self, fps=30, stop_seconds=4, flow_direction="down", crash_model_path="kaza_model.pt"):
        self.fps = fps
        self.stop_threshold_frames = int(fps * stop_seconds)
        self.flow_direction = flow_direction  
        self.track_history = defaultdict(lambda: deque(maxlen=self.stop_threshold_frames + 30))
        self.lane_counter = defaultdict(int) # Şerit ihlali için frame doğrulama sayacı
        self.lanes = []
        
        self.frame_count = 0
        self.last_crash_boxes = []
        
        self.crash_ai = None
        if os.path.exists(crash_model_path):
            self.crash_ai = YOLO(crash_model_path)
            print(f"[BILGI] {crash_model_path} basariyla Guvenlik Modulune entegre edildi.")
        else:
            print(f"[UYARI] {crash_model_path} bulunamadi. Kaza tespiti sadece hareket analiziyle yapilacak.")
        
    def add_lane(self, x1, y1, x2, y2):
        self.lanes.append((x1, y1, x2, y2))
        
    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_len_sq == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / line_len_sq))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return math.hypot(px - proj_x, py - proj_y)

    def analyze(self, frame, tracking_results):
        violations = []
        annotated_frame = frame.copy()
        h_orig, w_orig = frame.shape[:2]
        
        for (lx1, ly1, lx2, ly2) in self.lanes:
            cv2.line(annotated_frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2, cv2.LINE_AA)

        self.frame_count += 1

        if self.crash_ai is not None:
            if self.frame_count % 5 == 0:
                roi_ymin, roi_ymax = int(h_orig * 0.10), int(h_orig * 0.95)
                roi = frame[roi_ymin:roi_ymax, :]

                try:
                    results = self.crash_ai.predict(
                        roi,
                        device=0,
                        half=True,
                        conf=0.22,
                        imgsz=960,
                        max_det=5,
                        verbose=False
                    )
                    
                    self.last_crash_boxes = []
                    for r in results:
                        for box in r.boxes:
                            if box.conf[0] < 0.22:
                                continue
                            
                            crash_cls = self.crash_ai.names[int(box.cls[0])].lower()
                            if "kaza" not in crash_cls and "accident" not in crash_cls:
                                continue

                            cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
                            cy1 += roi_ymin
                            cy2 += roi_ymin

                            self.last_crash_boxes.append({
                                'cls': crash_cls,
                                'bbox': [cx1, cy1, cx2, cy2]
                            })
                except Exception:
                    pass

            for crash_item in self.last_crash_boxes:
                cx1, cy1, cx2, cy2 = crash_item['bbox']
                crash_cls = crash_item['cls']

                violations.append({'id': 'YAPAY_ZEKA', 'type': f'KAZA ({crash_cls.upper()})', 'bbox': [cx1, cy1, cx2, cy2]})
                
                cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)
                status_text = f"[KAZA TESPITI: {crash_cls.upper()}!]"
                (t_w, t_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (cx1, cy1 - t_h - 10), (cx1 + t_w, cy1), (0, 0, 0), -1)
                cv2.putText(annotated_frame, status_text, (cx1, cy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for obj in tracking_results:
            t_id = obj['id']
            x1, y1, x2, y2 = map(int, obj['bbox'])
            cls_name = obj.get('class', 'Araba')
            
            raw_center_x = (x1 + x2) // 2
            raw_center_y = (y1 + y2) // 2
            bottom_y = y2
            
            # Track Jitter önleme (Koordinatları yumuşatma)
            history = self.track_history[t_id]
            if history:
                prev_x, prev_y, _ = history[-1]
                center_x = int(prev_x * 0.7 + raw_center_x * 0.3)
                center_y = int(prev_y * 0.7 + raw_center_y * 0.3)
            else:
                center_x, center_y = raw_center_x, raw_center_y

            history.append((center_x, center_y, bottom_y))
            
            is_stopped = False
            is_wrong_way = False
            is_lane_violation = False
            
            # Dinamik Duraklama Eşiği (Kutunun yüksekliğine göre esnek hesaplama)
            box_h = y2 - y1
            stop_threshold = max(4, box_h * 0.20)

            if len(history) >= self.stop_threshold_frames:
                first_pos = history[-self.stop_threshold_frames]
                curr_pos = history[-1]
                dist = math.hypot(curr_pos[0] - first_pos[0], curr_pos[1] - first_pos[1])
                
                if dist < stop_threshold:  
                    is_stopped = True
                    violations.append({'id': t_id, 'type': 'DURAKLAMA', 'bbox': [x1, y1, x2, y2]})

            if len(history) >= 30: 
                past_y = history[-30][1]
                curr_y = history[-1][1]
                dy = curr_y - past_y
                
                if self.flow_direction == "down" and dy < -30:
                    is_wrong_way = True
                    violations.append({'id': t_id, 'type': 'TERS YON', 'bbox': [x1, y1, x2, y2]})
                elif self.flow_direction == "up" and dy > 30:
                    is_wrong_way = True
                    violations.append({'id': t_id, 'type': 'TERS YON', 'bbox': [x1, y1, x2, y2]})

            # Şerit İhlali - Çoklu Frame Doğrulama Sayacı ile
            lane_violated_in_this_frame = False
            for (lx1, ly1, lx2, ly2) in self.lanes:
                d = self._point_to_line_distance(center_x, bottom_y, lx1, ly1, lx2, ly2)
                if d < 5:  
                    lane_violated_in_this_frame = True
                    break

            if lane_violated_in_this_frame:
                self.lane_counter[t_id] += 1
            else:
                self.lane_counter[t_id] = 0

            if self.lane_counter[t_id] >= 5:
                is_lane_violation = True
                violations.append({'id': t_id, 'type': 'SERIT IHLALI', 'bbox': [x1, y1, x2, y2]})

            status_text = f"ID:{t_id} {cls_name}"
            box_color = None
            
            if is_stopped:
                box_color = (0, 0, 255) 
                status_text += " [DURAKLAMA!]"
            elif is_wrong_way:
                box_color = (0, 0, 255) 
                status_text += " [TERS YON!]"
            elif is_lane_violation:
                box_color = (0, 165, 255) 
                status_text += " [SERIT IHLALI!]"

            if box_color is not None:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 4)
                (t_w, t_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - t_h - 25), (x1 + t_w, y1 - 10), (0, 0, 0), -1)
                cv2.putText(annotated_frame, status_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        return annotated_frame, violations