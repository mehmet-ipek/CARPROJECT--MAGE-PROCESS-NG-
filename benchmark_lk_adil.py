import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import datetime

from src.yolo_detector import YoloCarDetector
from src.optical_flow import OpticalFlowTracker

LOG_DIR = "benchmark_logs_lk"
os.makedirs(LOG_DIR, exist_ok=True)

class LKWrapper:
    """
    LK optik akışı kullanarak çoklu nesne takibi yapar.
    Arayüzü ByteTrackWrapper ile aynıdır: update(frame) -> {track_id: (x,y,w,h,cls,conf)}
    """
    def __init__(self, detector):
        self.detector = detector
        self.trackers = {}          # track_id -> OpticalFlowTracker
        self.boxes = {}             # track_id -> (x,y,w,h)
        self.disappeared = {}       # track_id -> kayıp sayacı
        self.next_id = 1
        self.max_disappeared = 10   # 10 frame kaybolursa sil

    def update(self, frame):
        # 1. Modelin ham tespitlerini al (maske zaten dedektör içinde uygulanıyor)
        detections = self.detector.detect(frame)   # [(x,y,w,h,cls,conf), ...]

        # 2. Mevcut LK tracker'larını güncelle
        for tid in list(self.trackers.keys()):
            tracker = self.trackers[tid]
            success = tracker.track(frame)
            if success:
                self.boxes[tid] = tracker.get_box()
                self.disappeared[tid] = 0
            else:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.trackers[tid]
                    del self.boxes[tid]
                    del self.disappeared[tid]

        # 3. Yeni tespitleri mevcut kutularla eşleştir (basit merkez mesafesi)
        for det in detections:
            x, y, w, h, cls, conf = det
            cx = x + w / 2
            cy = y + h / 2

            # En yakın mevcut kutuyu bul
            min_dist = 50  # piksel cinsinden eşik
            matched_tid = None
            for tid, box in self.boxes.items():
                bx, by, bw, bh = box
                bcx = bx + bw / 2
                bcy = by + bh / 2
                dist = np.hypot(cx - bcx, cy - bcy)
                if dist < min_dist:
                    min_dist = dist
                    matched_tid = tid

            if matched_tid is None:
                # Yeni nesne başlat
                tracker = OpticalFlowTracker()
                if tracker.initialize(frame, (x, y, w, h)) is not None:
                    self.trackers[self.next_id] = tracker
                    self.boxes[self.next_id] = (x, y, w, h)
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1
            else:
                # Kutuyu yeni tespitle güncelle (opsiyonel: LK sonucunu da koruyabiliriz)
                # Burada LK'nın verdiği kutuyu güncel tespitle harmanlayalım (basitçe tespit kutusunu kabul edelim)
                # Ancak LK takipçisi zaten optik akışla güncellendi, karışıklık olmasın diye dokunmuyoruz.
                pass

        # 4. Sonuçları derle
        tracked_objects = {}
        for tid, box in self.boxes.items():
            if tid not in self.trackers:
                continue
            bx, by, bw, bh = box
            # Bu objeye en yakın tespiti bularak sınıf ve güven skoru ekle
            cx = bx + bw / 2
            cy = by + bh / 2
            best_cls = 0
            best_conf = 0.0
            min_dist = 1e9
            for det in detections:
                dx1, dy1, dw, dh, dcls, dconf = det
                dcx = dx1 + dw / 2
                dcy = dy1 + dh / 2
                dist = (cx - dcx) ** 2 + (cy - dcy) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_cls = dcls
                    best_conf = dconf
            tracked_objects[tid] = (int(bx), int(by), int(bw), int(bh), best_cls, best_conf)

        return tracked_objects

def benchmark_model(detector, model_name, video_path, max_frames=300):
    lk_wrapper = LKWrapper(detector)
    log_path = os.path.join(LOG_DIR, f"{model_name.replace(' ', '_')}.log")

    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Model: {model_name}\n")
        log_file.write(f"Tarih: {datetime.datetime.now()}\n")
        log_file.write("="*80 + "\n")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_boxes = 0
        unique_ids = set()
        track_class_history = {}
        fps_values = []

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            tracked = lk_wrapper.update(frame)
            elapsed = time.perf_counter() - start
            if elapsed > 0:
                fps_values.append(1.0 / elapsed)

            log_file.write(f"\n### Frame {frame_count+1} ###\n")
            log_file.write(f"Tespit sayısı: {len(tracked)}\n")

            for track_id, box_data in tracked.items():
                x1, y1, bw, bh, cls_id, conf = box_data
                unique_ids.add(track_id)
                total_boxes += 1

                if track_id not in track_class_history:
                    track_class_history[track_id] = Counter()
                track_class_history[track_id][cls_id] += 1

                log_file.write(f"  ID:{track_id} | cls:{cls_id} | conf:{conf:.3f} | "
                               f"box=({x1},{y1},{bw},{bh})\n")

            frame_count += 1

        cap.release()

        # Her track için baskın sınıfı bul
        final_class_counts = Counter()
        for tid, counter in track_class_history.items():
            dominant_cls = counter.most_common(1)[0][0]
            final_class_counts[dominant_cls] += 1

        log_file.write("\n" + "="*80 + "\n")
        log_file.write("ÖZET\n")
        log_file.write(f"Frame: {frame_count}\n")
        log_file.write(f"Benzersiz araç: {len(unique_ids)}\n")
        log_file.write(f"Toplam kutu: {total_boxes}\n")
        avg_fps = np.mean(fps_values) if fps_values else 0
        log_file.write(f"Ortalama FPS: {avg_fps:.2f}\n")
        class_names = {0:'Araba',1:'Motor',2:'Otobus',3:'Kamyonet'}
        log_file.write("Sınıf dağılımı (benzersiz araçlar):\n")
        for cls_id, count in sorted(final_class_counts.items()):
            log_file.write(f"  {class_names.get(cls_id, f'ID:{cls_id}')}: {count}\n")

    return {
        'frames': frame_count,
        'unique_vehicles': len(unique_ids),
        'total_boxes': total_boxes,
        'class_counts': dict(final_class_counts),
        'avg_fps': avg_fps,
        'log_path': log_path
    }

def plot_results(results):
    models = list(results.keys())
    unique_vals = [res['unique_vehicles'] for res in results.values()]
    box_vals = [res['total_boxes'] for res in results.values()]
    fps_vals = [res['avg_fps'] for res in results.values()]

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    bars = plt.bar(models, unique_vals, color='steelblue')
    plt.title('Benzersiz Araç Sayısı (LK Optik Akış)')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                 ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 2, 2)
    plt.bar(models, box_vals, color='seagreen')
    plt.title('Toplam Kutu Sayısı (LK)')

    plt.subplot(2, 2, 3)
    plt.bar(models, fps_vals, color='darkorange')
    plt.title('Ortalama FPS (LK)')

    plt.subplot(2, 2, 4)
    all_classes = sorted(set().union(*[res['class_counts'].keys() for res in results.values()]))
    bottom = np.zeros(len(models))
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
    class_names = {0:'Araba',1:'Motor',2:'Otobus',3:'Kamyonet'}
    for i, cls in enumerate(all_classes):
        counts = [res['class_counts'].get(cls, 0) for res in results.values()]
        plt.bar(models, counts, bottom=bottom,
                label=class_names.get(cls, f'ID:{cls}'),
                color=colors[i % len(colors)])
        bottom += np.array(counts)
    plt.title('Sınıf Dağılımı (LK)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('benchmark_lk_adil.png', dpi=150)
    plt.close()
    print("\n✅ Grafik 'benchmark_lk_adil.png' olarak kaydedildi.")

def main():
    video_path = "trafikvideo.mp4"
    max_frames = 300

    models = {
        "Custom 50ep (best.pt)": YoloCarDetector("best.pt"),
        "YOLOv8n": YoloCarDetector("yolov8n.pt"),
        "YOLOv8s": YoloCarDetector("yolov8s.pt"),
        "YOLO11n": YoloCarDetector("yolo11n.pt"),
        "YOLO11s": YoloCarDetector("yolo11s.pt"),
    }

    results = {}
    for name, det in models.items():
        print(f"\n--- {name} test ediliyor (LK Optik Akış) ---")
        res = benchmark_model(det, name, video_path, max_frames)
        results[name] = res
        print(f"  Benzersiz: {res['unique_vehicles']}, Kutu: {res['total_boxes']}, "
              f"FPS: {res['avg_fps']:.1f}")

    print("\n" + "="*80)
    print(f"{'Model':<25} {'Benzersiz':<10} {'Toplam Kutu':<12} {'FPS':<8}")
    print("-"*80)
    for name, res in results.items():
        print(f"{name:<25} {res['unique_vehicles']:<10} {res['total_boxes']:<12} {res['avg_fps']:<8.1f}")

    class_names = {0:'Araba',1:'Motor',2:'Otobus',3:'Kamyonet'}
    print("\n--- Sınıf Dağılımı (Benzersiz Araçlar) ---")
    for name, res in results.items():
        print(f"\n{name}:")
        for cls_id, count in sorted(res['class_counts'].items()):
            print(f"  {class_names.get(cls_id, f'ID:{cls_id}')}: {count} adet")

    plot_results(results)
    print(f"\nLog dosyaları '{LOG_DIR}' klasöründe.")

if __name__ == "__main__":
    main()