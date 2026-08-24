import csv
import io
import os
import time
import math
import threading
import random
from collections import Counter
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify

from src.camera import Camera
from src.yolo_detector import YoloCarDetector
from src.rtdetr_detector import RTDetrDetector
from src.mobilenet_detector import MobileNetDetector
from src.optical_flow import OpticalFlowTracker
from src.csrt_tracker import CSRTTracker
from src.kcf_tracker import KCFTracker
from src.mosse_tracker import MOSSETracker
from src.bytetrack_tracker import ByteTrackWrapper
from src.deepsort_tracker import DeepSortWrapper
from src.statistics import Statistics
from src.database import DatabaseManager
from src.centroid_tracker import CentroidTracker
from src.analytics import TrafficHeatmap
from src.color_detector import get_dominant_color

app = Flask(__name__)
db = DatabaseManager()

CURRENT_VIDEO_NAME = "trafikvideo.mp4"
EXPORT_DIR = "otomatik_kirpilmis_araclar_trafikvideo"
os.makedirs(EXPORT_DIR, exist_ok=True)

DETECTOR_TYPE = "YOLO_CUSTOM"
TRACKER_TYPE = "BYTETRACK"
DISTANCE_TYPE = "CHEBYSHEV"
DETECTION_INTERVAL = 5
MIN_WIDTH = 2
MIN_HEIGHT = 2

config_lock = threading.Lock()

camera = Camera(video_path=CURRENT_VIDEO_NAME)
stats = Statistics()
heatmap = TrafficHeatmap()

ct = CentroidTracker(max_disappeared=50, max_distance=60)
total_cars_counted = 0

class_counts = {"Araba": 0, "Motor": 0, "Otobus": 0, "Kamyonet": 0}

trackable_objects = {}
motion_history = {}

correct_preds = 0
wrong_preds = 0
last_result_text = "Arac Gecisi Bekleniyor..."
last_result_color = (0, 255, 255)
smoothed_accuracy = 0.0

detector = None
tracker = None
bytetrack = None
deepsort = None

def save_image_async(folder_path, file_name, image):
    try:
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, file_name)
        cv2.imwrite(save_path, image)
    except Exception as e:
        print(f"[HATA] Resim yazılamadı: {e}")

def load_models():
    global detector, tracker, bytetrack, deepsort

    if DETECTOR_TYPE == "YOLO_CUSTOM":
        detector = YoloCarDetector("best.pt")
    elif DETECTOR_TYPE == "RT-DETR":
        detector = RTDetrDetector("rtdetr-l.pt")
    elif DETECTOR_TYPE == "MobileNet":
        detector = MobileNetDetector()
    elif DETECTOR_TYPE == "YOLOv8n":
        detector = YoloCarDetector("yolov8n.pt")
    elif DETECTOR_TYPE == "YOLOv8s":
        detector = YoloCarDetector("yolov8s.pt")
    elif DETECTOR_TYPE == "YOLO11n":
        detector = YoloCarDetector("yolo11n.pt")
    else:
        detector = YoloCarDetector("yolo11s.pt")

    bytetrack = None
    deepsort = None
    if DETECTOR_TYPE.startswith("YOLO"):
        try:
            bytetrack = ByteTrackWrapper(detector)
            deepsort = DeepSortWrapper(detector)
        except Exception as e:
            print(f"[UYARI] AI Tracker başlatılamadı: {e}")

    max_w, max_h = 1920, 1080

    if TRACKER_TYPE == "LK":
        tracker = OpticalFlowTracker()
    elif TRACKER_TYPE == "CSRT":
        tracker = CSRTTracker()
    elif TRACKER_TYPE == "KCF":
        tracker = KCFTracker()
    elif TRACKER_TYPE == "MOSSE":
        tracker = MOSSETracker()
    elif TRACKER_TYPE in ["BYTETRACK", "DEEPSORT"]:
        tracker = None

    return max_w, max_h

MAX_WIDTH, MAX_HEIGHT = load_models()

def get_unique_color(track_id):
    random.seed(int(track_id) * 42)
    b = random.randint(50, 255)
    g = random.randint(50, 255)
    r = random.randint(50, 255)
    return (b, g, r)

def filter_objects(objects, w_frame, h_frame, current_mask=None):
    filtered = []
    for obj in objects:
        x, y, w, h = obj[:4]
        cls_id = obj[4] if len(obj) > 4 else 0

        if w >= MIN_WIDTH and h >= MIN_HEIGHT and w <= MAX_WIDTH and h <= MAX_HEIGHT:
            if current_mask is not None:
                cx = int(x + w / 2)
                cy = int(y + h / 2)
                if 0 <= cx < w_frame and 0 <= cy < h_frame:
                    if current_mask[cy, cx] == 0:
                        continue
            filtered.append((x, y, w, h, cls_id))
    return filtered

def get_class_name(cls_id):
    if hasattr(detector, 'model') and hasattr(detector.model, 'names'):
        raw_name = detector.model.names.get(cls_id, None)
        if raw_name:
            turkce_map = {
                "cars": "Araba",
                "car": "Araba",
                "motorcycle": "Motor",
                "bus": "Otobus",
                "truck": "Kamyonet"
            }
            return turkce_map.get(raw_name.lower(), raw_name)

    yedek_map = {0: "Araba", 1: "Motor", 2: "Otobus", 3: "Kamyonet"}
    return yedek_map.get(cls_id, "Bilinmeyen")

def generate_frames():
    global DETECTOR_TYPE, TRACKER_TYPE, DISTANCE_TYPE, stats
    global total_cars_counted, trackable_objects, motion_history, class_counts
    global correct_preds, wrong_preds, last_result_text, last_result_color, smoothed_accuracy

    frame_number = 0
    fps_values = []
    cached_detections = []

    mask_raw = cv2.imread("trafikvideo.jpg", cv2.IMREAD_GRAYSCALE)
    if mask_raw is not None:
        _, mask_bin = cv2.threshold(mask_raw, 100, 255, cv2.THRESH_BINARY)
    else:
        mask_bin = None

    while True:
        try:
            frame_start = time.time()
            frame = camera.get_frame()

            if frame is None:
                break

            frame_number += 1
            current_time = time.time()
            h, w = frame.shape[:2]

            # Maskeyi gerçekten uygula
            if mask_bin is not None:
                current_mask = cv2.resize(mask_bin, (w, h))
            else:
                current_mask = np.full((h, w), 255, dtype=np.uint8)

            zone_start = int(h * 0.50)
            zone_end = int(h * 0.70)
            COUNT_MARGIN = 15

            objects_dict = {}
            detected_objects = []
            active_objects_count = 0
            distance = 0.0
            iou = 0.0

            with config_lock:
                current_tracker_type = TRACKER_TYPE
                current_detector_type = DETECTOR_TYPE
                current_distance_type = DISTANCE_TYPE

            if current_tracker_type in ["BYTETRACK", "DEEPSORT"]:
                raw_tracked = {}
                if current_tracker_type == "BYTETRACK" and bytetrack is not None:
                    raw_tracked = bytetrack.update(frame)
                elif current_tracker_type == "DEEPSORT" and deepsort is not None:
                    raw_tracked = deepsort.update(frame)

                tracked_bboxes = {}
                for t_id, box_data in raw_tracked.items():
                    x1, y1, bw, bh = box_data[:4]
                    cx = int(x1 + bw / 2)
                    cy = int(y1 + bh / 2)

                    if not (0 <= cx < w and 0 <= cy < h):
                        continue

                    tracked_bboxes[t_id] = box_data

                heatmap.update_from_mot({t: b[:4] for t, b in tracked_bboxes.items()})

                for t_id, box_data in list(tracked_bboxes.items()):
                    x1, y1, bw, bh = box_data[:4]
                    c_x1, c_y1 = max(0, int(x1)), max(0, int(y1))
                    c_x2, c_y2 = min(w, int(x1 + bw)), min(h, int(y1 + bh))
                    cropped_vehicle = frame[c_y1:c_y2, c_x1:c_x2].copy()

                    cx = int(x1 + bw / 2)
                    cy = int(y1 + bh / 2)

                    objects_dict[t_id] = (cx, cy)
                    detected_objects.append(box_data)
                    active_objects_count += 1

                    det_cls_id = int(box_data[4]) if len(box_data) > 4 else 0
                    raw_class = get_class_name(det_cls_id)

                    if t_id not in trackable_objects:
                        trackable_objects[t_id] = {
                            "path": [(cx, cy)],
                            "counted": False,
                            "count_time": 0.0,
                            "current_class": raw_class,
                            "class_history": [raw_class]
                        }
                    else:
                        track = trackable_objects[t_id]
                        track["path"].append((cx, cy))
                        if len(track["path"]) > 50:
                            track["path"].pop(0)
                        track["class_history"].append(raw_class)
                        if len(track["class_history"]) > 10:
                            track["class_history"].pop(0)
                        most_common = Counter(track["class_history"]).most_common(1)[0][0]
                        track["current_class"] = most_common

                    track = trackable_objects[t_id]
                    path = track["path"]
                    counted = track["counted"]
                    current_class = track.get("current_class", "Araba")
                    prev_y = path[-2][1] if len(path) > 1 else cy

                    if not counted and (prev_y < zone_end + COUNT_MARGIN and cy >= zone_end - COUNT_MARGIN):
                        total_cars_counted += 1
                        if current_class not in class_counts:
                            class_counts[current_class] = 0
                        class_counts[current_class] += 1
                        track["counted"] = True
                        track["count_time"] = current_time

                        history = track["class_history"]
                        if len(history) >= 5:
                            most_common_cls, freq = Counter(history).most_common(1)[0]
                            consistency = freq / len(history)
                        else:
                            most_common_cls = current_class
                            consistency = 1.0

                        if consistency >= 0.70:
                            correct_preds += 1
                            last_result_text = f"Basarili Eslesme: [{most_common_cls}] (%{consistency*100:.0f})"
                            last_result_color = (0, 255, 0)
                        else:
                            wrong_preds += 1
                            last_result_text = f"Kararsiz Eslesme: [{most_common_cls}] (%{consistency*100:.0f})"
                            last_result_color = (0, 0, 255)

                        if cropped_vehicle is not None and cropped_vehicle.size > 0:
                            target_subfolder = os.path.join(EXPORT_DIR, current_class)
                            detected_color = get_dominant_color(cropped_vehicle)
                            resized_vehicle = cv2.resize(cropped_vehicle, (300, 300), interpolation=cv2.INTER_CUBIC)
                            cv2.putText(resized_vehicle, f"TAHMIN: {current_class} ({detected_color})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            file_name = f"frame_{frame_number}_id_{t_id}_{detected_color}.jpg"
                            threading.Thread(target=save_image_async, args=(target_subfolder, file_name, resized_vehicle)).start()

                    path_color = get_unique_color(t_id)
                    for i in range(1, len(path)):
                        thickness = int(math.sqrt(64 / float(i + 1)) * 1.5)
                        cv2.line(frame, path[i - 1], path[i], path_color, thickness)

                    box_color = path_color
                    id_text = f"ID:{t_id} {current_class}"

                    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), box_color, 3)
                    (t_w, t_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - t_h - 10), (x1 + t_w, y1), (0, 0, 0), -1)
                    cv2.putText(frame, id_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

            else:
                if current_detector_type in ["RT-DETR", "MobileNet"]:
                    if frame_number % 3 == 0 or not cached_detections:
                        cached_detections = filter_objects(detector.detect(frame), w, h, current_mask)
                    detected_objects = cached_detections
                else:
                    detected_objects = filter_objects(detector.detect(frame), w, h, current_mask)

                rects_for_ct = [obj[:4] for obj in detected_objects]
                cls_ids_for_ct = {i: obj[4] for i, obj in enumerate(detected_objects)}
                conf_for_ct = {i: obj[5] if len(obj) > 5 else 1.0 for i, obj in enumerate(detected_objects)}

                objects_dict = ct.update(rects_for_ct)
                heatmap.update_from_centroid(objects_dict, ct.boxes)

                for object_id, centroid in objects_dict.items():
                    if ct.disappeared.get(object_id, 0) > 0:
                        continue

                    cx, cy = centroid
                    obj_box = ct.boxes.get(object_id)
                    active_objects_count += 1

                    cropped_vehicle = None
                    final_class = "Araba"

                    if obj_box is not None:
                        bx, by, bw, bh = obj_box
                        c_x1, c_y1 = max(0, int(bx)), max(0, int(by))
                        c_x2, c_y2 = min(w, int(bx + bw)), min(h, int(by + bh))
                        cropped_vehicle = frame[c_y1:c_y2, c_x1:c_x2].copy()

                        for i, det_box in enumerate(rects_for_ct):
                            if det_box == obj_box:
                                det_cls_id = cls_ids_for_ct[i]
                                final_class = get_class_name(det_cls_id)
                                break

                    if object_id not in trackable_objects:
                        trackable_objects[object_id] = {
                            "path": [(cx, cy)],
                            "counted": False,
                            "count_time": 0.0,
                            "current_class": final_class,
                            "class_history": [final_class]
                        }
                    else:
                        track = trackable_objects[object_id]
                        track["path"].append((cx, cy))
                        if len(track["path"]) > 20:
                            track["path"].pop(0)
                        track["class_history"].append(final_class)
                        if len(track["class_history"]) > 10:
                            track["class_history"].pop(0)
                        most_common = Counter(track["class_history"]).most_common(1)[0][0]
                        track["current_class"] = most_common

                    track = trackable_objects[object_id]
                    counted = track["counted"]
                    path = track["path"]
                    current_class = track.get("current_class", "Araba")
                    prev_y = path[-2][1] if len(path) > 1 else cy

                    if not counted and (prev_y < zone_end + COUNT_MARGIN and cy >= zone_end - COUNT_MARGIN):
                        total_cars_counted += 1
                        if current_class not in class_counts:
                            class_counts[current_class] = 0
                        class_counts[current_class] += 1
                        track["counted"] = True
                        track["count_time"] = current_time

                        history = track["class_history"]
                        if len(history) >= 5:
                            most_common_cls, freq = Counter(history).most_common(1)[0]
                            consistency = freq / len(history)
                        else:
                            most_common_cls = current_class
                            consistency = 1.0

                        if consistency >= 0.70:
                            correct_preds += 1
                            last_result_text = f"Basarili Eslesme: [{most_common_cls}] (%{consistency*100:.0f})"
                            last_result_color = (0, 255, 0)
                        else:
                            wrong_preds += 1
                            last_result_text = f"Kararsiz Eslesme: [{most_common_cls}] (%{consistency*100:.0f})"
                            last_result_color = (0, 0, 255)

                        if cropped_vehicle is not None and cropped_vehicle.size > 0:
                            target_subfolder = os.path.join(EXPORT_DIR, current_class)
                            detected_color = get_dominant_color(cropped_vehicle)
                            resized_vehicle = cv2.resize(cropped_vehicle, (300, 300), interpolation=cv2.INTER_CUBIC)
                            cv2.putText(resized_vehicle, f"TAHMIN: {current_class} ({detected_color})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            file_name = f"frame_{frame_number}_id_{object_id}_{detected_color}.jpg"
                            threading.Thread(target=save_image_async, args=(target_subfolder, file_name, resized_vehicle)).start()

                    for i in range(1, len(path)):
                        thickness = int(math.sqrt(64 / float(i + 1)) * 1.5)
                        cv2.line(frame, path[i - 1], path[i], (255, 0, 0), thickness)

                    if obj_box is not None:
                        bx, by, bw, bh = obj_box
                        box_color = (0, 255, 255)
                        id_text = f"ID:{object_id} {current_class}"

                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 3)
                        (t_w, t_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(frame, (bx, by - t_h - 10), (bx + t_w, by), (0, 0, 0), -1)
                        cv2.putText(frame, id_text, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

            active_ids = set(objects_dict.keys())
            for object_id in list(trackable_objects.keys()):
                if object_id not in active_ids:
                    del trackable_objects[object_id]

            if frame_number % 3 == 0:
                frame = heatmap.get_colored_heatmap(frame)

            cv2.line(frame, (0, zone_start), (w, zone_start), (0, 165, 255), 2)
            cv2.putText(frame, "Giris Cizgisi", (10, zone_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            cv2.line(frame, (0, zone_end), (w, zone_end), (0, 0, 255), 2)
            cv2.putText(frame, "Sayim Cizgisi", (10, zone_end - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if active_objects_count <= 4:
                density_status = "AKICI"
                density_color = (0, 255, 0)
            elif active_objects_count <= 8:
                density_status = "ORTA YOGUN"
                density_color = (0, 255, 255)
            else:
                density_status = "SIKISIK"
                density_color = (0, 0, 255)

            total_preds = correct_preds + wrong_preds
            if total_preds > 0:
                smoothed_accuracy = (correct_preds / total_preds) * 100.0
            else:
                smoothed_accuracy = 0.0

            overlay = frame.copy()
            box_width = 420
            box_height = 40 + (len(class_counts) * 20) + 80

            cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, f"TOPLAM GECEN: {total_cars_counted}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            y_offset = 60
            for cls_name, count in class_counts.items():
                cv2.putText(frame, f"- {cls_name}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_offset += 20

            cv2.putText(frame, f"EKRANDAKI ARAC: {active_objects_count}", (20, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"TRAFIK: {density_status}", (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, density_color, 2)

            cv2.putText(frame, f"SISTEM BASARISI: %{smoothed_accuracy:.1f} ({correct_preds}/{total_preds})", (20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"SONUC: {last_result_text}", (20, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, last_result_color, 2)

            frame_time = time.time() - frame_start
            fps = 1 / frame_time if frame_time > 0 else 0
            fps_values.append(fps)
            if len(fps_values) > 30:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            db.log_data(frame_number, current_detector_type, current_tracker_type, current_distance_type, distance, iou, avg_fps)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"[HATA] Beklenmeyen hata olustu, kare atlaniyor: {e}")
            continue

@app.route('/')
def index():
    with config_lock:
        det_t = DETECTOR_TYPE
        trk_t = TRACKER_TYPE
        dst_t = DISTANCE_TYPE
    logs = db.get_latest_logs(10)
    return render_template('index.html', logs=logs, detector_type=det_t, tracker_type=trk_t, distance_type=dst_t)

@app.route('/update_config', methods=['POST'])
def update_config():
    global DETECTOR_TYPE, TRACKER_TYPE, DISTANCE_TYPE, MAX_WIDTH, MAX_HEIGHT
    global trackable_objects, motion_history, ct, heatmap, total_cars_counted, class_counts
    global correct_preds, wrong_preds, last_result_text, last_result_color, smoothed_accuracy

    with config_lock:
        DETECTOR_TYPE = request.form.get('detector')
        TRACKER_TYPE = request.form.get('tracker')
        DISTANCE_TYPE = request.form.get('distance')
        MAX_WIDTH, MAX_HEIGHT = load_models()

        total_cars_counted = 0
        class_counts = {"Araba": 0, "Motor": 0, "Otobus": 0, "Kamyonet": 0}

        correct_preds = 0
        wrong_preds = 0
        smoothed_accuracy = 0.0
        last_result_text = "Arac Gecisi Bekleniyor..."
        last_result_color = (0, 255, 255)

        trackable_objects.clear()
        motion_history.clear()
        ct = CentroidTracker(max_disappeared=50, max_distance=60)
        heatmap.reset()

    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    global class_counts
    return jsonify(class_counts)

@app.route('/download_csv')
def download_csv():
    logs = db.get_all_logs()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Tarih', 'Frame', 'Dedektor', 'Tracker', 'Mesafe Tipi', 'Mesafe (px)', 'IOU', 'FPS'])
    cw.writerows(logs)

    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=otonom_arac_sistem_raporu.csv"}
    )

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)