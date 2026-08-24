# src/mobilenet_detector.py
import cv2
import numpy as np
import os

class MobileNetDetector:
    def __init__(self, prototxt_path="deploy.prototxt", model_path="mobilenet_iter_73000.caffemodel"):
        self.min_width = 15
        self.min_height = 15
        self.net = None

        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("[BAŞARILI] MobileNet-SSD Caffe ağı yüklendi!")
            except Exception as e:
                print(f"[HATA] MobileNet yüklenemedi: {e}")
        else:
            print("[HATA] Model dosyaları eksik!")

        # Maske (trafikvideo.jpg)
        mask_raw = cv2.imread("trafikvideo.jpg", cv2.IMREAD_GRAYSCALE)
        if mask_raw is not None:
            _, self.mask_bin = cv2.threshold(mask_raw, 100, 255, cv2.THRESH_BINARY)
        else:
            self.mask_bin = None

    def detect(self, frame):
        if self.net is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        current_mask = (
            cv2.resize(self.mask_bin, (frame_w, frame_h))
            if self.mask_bin is not None
            else np.full((frame_h, frame_w), 255, dtype=np.uint8)
        )

        boxes = []
        try:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.25:
                    idx = int(detections[0, 0, i, 1])

                    cls_mapping = {7: 0,   # car → cars (0)
                                   6: 2,   # bus → bus (2)
                                   14: 3}  # train → truck (3)
                    if idx not in cls_mapping:
                        continue

                    mapped_cls_id = cls_mapping[idx]
                    box = detections[0, 0, i, 3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
                    x1, y1, x2, y2 = box.astype("int")
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)

                    w_box, h_box = x2 - x1, y2 - y1
                    if w_box < self.min_width or h_box < self.min_height:
                        continue

                    cx, cy = int(x1 + w_box / 2), int(y1 + h_box / 2)
                    if 0 <= cx < frame_w and 0 <= cy < frame_h:
                        if current_mask[cy, cx] == 0:
                            continue
                    else:
                        continue

                    boxes.append((x1, y1, w_box, h_box, mapped_cls_id, float(confidence)))
        except Exception as e:
            print(f"[MobileNet] Tespit hatası: {e}")

        return boxes