# src/rtdetr_detector.py
from ultralytics import RTDETR
import numpy as np
import cv2

class RTDetrDetector:
    def __init__(self, model_path="rtdetr-l.pt"):
        self.model = RTDETR(model_path)
        self.allowed_classes = [2, 3, 5, 7]  
        self.min_width = 20
        self.min_height = 20
        
        # trafikvideo.jpg
        self.mask_raw = cv2.imread("trafikvideo.jpg", cv2.IMREAD_GRAYSCALE)
        self.mask_bin = cv2.threshold(self.mask_raw, 127, 255, cv2.THRESH_BINARY)[1] if self.mask_raw is not None else None
        
    def detect(self, frame):
        h, w = frame.shape[:2]
        current_mask = cv2.resize(self.mask_bin, (w, h)) if self.mask_bin is not None else None

        results = self.model(frame, verbose=False, classes=self.allowed_classes, imgsz=512, conf=0.4)
        boxes = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy() 
                x1, y1, x2, y2 = map(int, coords)
                
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                bx = int(x1 + (x2 - x1) / 2)
                by = int(y1 + (y2 - y1) / 2)
                
                if current_mask is not None:
                    if 0 <= bx < w and 0 <= by < h:
                        if current_mask[by, cx := bx] == 0:
                            continue 

                w_box = x2 - x1
                h_box = y2 - y1
                if w_box >= self.min_width and h_box >= self.min_height:
                    boxes.append((x1, y1, w_box, h_box, cls_id, conf))
                    
        return boxes