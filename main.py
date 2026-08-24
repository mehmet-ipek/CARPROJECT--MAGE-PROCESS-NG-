from src.camera import Camera
from src.yolo_detector import YoloCarDetector
from src.optical_flow import OpticalFlowTracker
from src.csrt_tracker import CSRTTracker
from src.kcf_tracker import KCFTracker
from src.mosse_tracker import MOSSETracker
from src.statistics import Statistics

import cv2
import time
import os
import math

DETECTOR_TYPE = "YOLO"  
TRACKER_TYPE = "LK"
DISTANCE_TYPE = "CHEBYSHEV"
DETECTION_INTERVAL = 10

MIN_WIDTH = 40
MIN_HEIGHT = 40
MAX_WIDTH = 1920
MAX_HEIGHT = 1080

os.makedirs("output", exist_ok=True)
video_writer = None

def filter_objects(objects):
    filtered = []
    for (x, y, w, h) in objects:
        if w >= MIN_WIDTH and h >= MIN_HEIGHT and w <= MAX_WIDTH and h <= MAX_HEIGHT:
            filtered.append((x, y, w, h))
    return filtered

camera = Camera()
detector = YoloCarDetector("yolov8n.pt")

if TRACKER_TYPE == "LK":
    tracker = OpticalFlowTracker()
elif TRACKER_TYPE == "CSRT":
    tracker = CSRTTracker()
elif TRACKER_TYPE == "KCF":
    tracker = KCFTracker()
elif TRACKER_TYPE == "MOSSE":
    tracker = MOSSETracker()
else:
    raise Exception("Bilinmeyen tracker tipi: " + TRACKER_TYPE)

stats = Statistics()

frame_number = 0
initialized = False
fps_values = []
start_time = time.time()
selected_distances = []
iou_values = []

while True:
    frame = camera.get_frame()
    if frame is None:
        print("Kameradan görüntü alınamıyor.")
        break

    frame_start = time.time()
    frame_number += 1

    if video_writer is None:
        h, w, _ = frame.shape
        video_writer = cv2.VideoWriter(
            "output/car_tracking_result.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (w, h)
        )

    detected_objects = filter_objects(detector.detect(frame))

    for (x, y, ow, oh) in detected_objects:
        cv2.rectangle(frame, (x, y), (x+ow, y+oh), (0, 255, 255), 2)
        cv2.putText(frame, f"Detection {ow}x{oh}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if TRACKER_TYPE == "LK":
        if not initialized or frame_number % DETECTION_INTERVAL == 0:
            re_objects = filter_objects(detector.detect(frame))
            if len(re_objects) >= 1:
                result = tracker.initialize(frame, re_objects[0])
                if result is not None:
                    initialized = True
            else:
                initialized = False
    else:
        if not initialized:
            re_objects = filter_objects(detector.detect(frame))
            if len(re_objects) >= 1:
                result = tracker.initialize(frame, re_objects[0])
                if result is not None:
                    initialized = True

    box = None
    if initialized:
        if TRACKER_TYPE == "LK":
            result = tracker.track(frame)
            if result is not None:
                old_points, new_points = result
                if len(new_points) < 10:
                    initialized = False
                else:
                    stats.update(old_points, new_points)
                    frame = tracker.draw_tracks(frame, old_points, new_points)
        else:
            success = tracker.track(frame)
            if not success:
                initialized = False

    box = tracker.get_box()
    if box is not None and initialized:
        x, y, bw, bh = box
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
        cv2.putText(frame, f"{TRACKER_TYPE} Tracker", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    frame_time = time.time() - frame_start
    fps = 1 / frame_time if frame_time > 0 else 0
    fps_values.append(fps)
    avg_fps = sum(fps_values) / len(fps_values)

    cv2.imshow("Car Tracking System", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

try:
    camera.release()
except:
    pass 

if video_writer:
    video_writer.release()

cv2.destroyAllWindows()
print("İşlem tamamlandı.")