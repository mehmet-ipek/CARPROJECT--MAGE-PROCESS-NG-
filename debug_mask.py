import cv2
import numpy as np
from src.yolo_detector import YoloCarDetector

video_path = "trafikvideo.mp4"
max_frames = 30

# Maskeli dedektör (orijinal hali)
detector_masked = YoloCarDetector("best.pt")

# Maskesiz dedektör (geçici olarak maskeyi kaldıralım)
detector_unmasked = YoloCarDetector("best.pt")
detector_unmasked.mask_bin = None          # maskeyi iptal et

cap = cv2.VideoCapture(video_path)

masked_counts = []
unmasked_counts = []

for _ in range(max_frames):
    ret, frame = cap.read()
    if not ret:
        break

    det_masked = detector_masked.detect(frame)
    det_unmasked = detector_unmasked.detect(frame)

    masked_counts.append(len(det_masked))
    unmasked_counts.append(len(det_unmasked))

cap.release()

print(f"Maskeli ortalama tespit: {np.mean(masked_counts):.1f}")
print(f"Maskesiz ortalama tespit: {np.mean(unmasked_counts):.1f}")
print(f"Fark (maskeden dolayı kayıp): {np.mean(unmasked_counts) - np.mean(masked_counts):.1f} tespit/frame")