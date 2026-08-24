# benchmark_bytetrack.py
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import datetime

from src.yolo_detector import YoloCarDetector
from src.bytetrack_tracker import ByteTrackWrapper

LOG_DIR = "benchmark_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def benchmark_model_with_log(detector, model_name, video_path, max_frames=300):
    tracker = ByteTrackWrapper(detector)  # Güncel ByteTrack (maskeli, conf=0.05)

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
            tracked = tracker.update(frame)      # ByteTrack ile takip
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
    plt.title('Benzersiz Araç Sayısı')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                 ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 2, 2)
    plt.bar(models, box_vals, color='seagreen')
    plt.title('Toplam Kutu Sayısı')

    plt.subplot(2, 2, 3)
    plt.bar(models, fps_vals, color='darkorange')
    plt.title('Ortalama FPS')

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
    plt.title('Sınıf Dağılımı')
    plt.legend()

    plt.tight_layout()
    plt.savefig('benchmark_bytetrack_yeni.png', dpi=150)
    plt.close()
    print("\n✅ Grafik 'benchmark_bytetrack_yeni.png' olarak kaydedildi.")

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
        print(f"\n--- {name} test ediliyor ---")
        res = benchmark_model_with_log(det, name, video_path, max_frames)
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