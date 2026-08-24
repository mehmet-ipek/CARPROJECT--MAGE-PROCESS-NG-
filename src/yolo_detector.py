import cv2
import numpy as np
from ultralytics import YOLO

class YoloCarDetector:

    def __init__(self, model_path="best.pt"):
        print("=" * 60)
        print(f"[BILGI] Ozel YOLO Modeli Yukleniyor: {model_path}")

        self.model = YOLO(model_path)
        print(f"[MODEL SINIFLARI] {self.model.names}")

        self.last_objects = []
        self.allowed_classes = list(self.model.names.keys())

        # Min boyut iyice düşük
        self.min_width = 2
        self.min_height = 2

        print(f"[IZIN VERILEN CLASS ID] {self.allowed_classes}")
        print(f"[MIN BOYUT] Genislik={self.min_width}, Yukseklik={self.min_height}")
        print("=" * 60)

        # Maskeyi yükle (trafikvideo.jpg)
        self.mask_bin = None
        mask_raw = cv2.imread("trafikvideo.jpg", cv2.IMREAD_GRAYSCALE)
        if mask_raw is not None:
            _, self.mask_bin = cv2.threshold(mask_raw, 100, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            self.mask_bin = cv2.dilate(self.mask_bin, kernel, iterations=2)
            print("[BILGI] Maske yuklendi ve genisletildi: trafikvideo.jpg")
        else:
            print("[UYARI] trafikvideo.jpg bulunamadi. Maske devre disi.")

    def _get_dynamic_conf(self, frame):
        # Güveni artırılmış model için eşik 0.30
        return 0.10

    def detect(self, frame):
        conf_thresh = self._get_dynamic_conf(frame)

        results = self.model.predict(
            source=frame,
            conf=conf_thresh,
            imgsz=640,
            max_det=500,
            agnostic_nms=False,
            verbose=False
        )

        objects = []
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
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

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
                if not (0 <= cx < frame_w and 0 <= cy < frame_h):
                    continue

                if current_mask[cy, cx] == 0:
                    continue

                class_name = self.model.names.get(cls_id, str(cls_id))
                print(f"[YOLO DETECT] CLASS_ID={cls_id} | CLASS={class_name} | CONF={conf:.3f} | BOX=({x1},{y1},{bw},{bh})")

                objects.append((x1, y1, bw, bh, cls_id, conf))

        self.last_objects = objects
        return objects