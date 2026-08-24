import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from src.yolo_detector import YoloCarDetector
from src.optical_flow import OpticalFlowTracker

TARGET_CLASSES = {
    "car": "Araba",
    "cars": "Araba",
    "motorcycle": "Motor",
    "bus": "Otobus",
    "truck": "Kamyonet"
}

def get_class_name_for_model(model, cls_id):
    raw_name = model.names.get(int(cls_id), None)
    if raw_name is None:
        return None
    for eng, tr in TARGET_CLASSES.items():
        if raw_name.lower() == eng.lower():
            return tr
    return None

def benchmark_model_with_LK(detector, video_path, max_frames=500):
    cap = cv2.VideoCapture(video_path)
    tracker = OpticalFlowTracker()
    frame_count = 0
    total_detections = 0
    class_counter = Counter()
    total_conf = 0.0
    fps_values = []

    model = detector.model
    # Maskeyi olduğu gibi bırakıyoruz (maskeli çalışsın)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        detections = detector.detect(frame)

        # Her yeni tespiti sadece bir kez sayalım (LK zaten onları takip edecek)
        for det in detections:
            cls_id = int(det[4])
            conf = float(det[5])
            turkce_etiket = get_class_name_for_model(model, cls_id)
            if turkce_etiket is not None:
                total_detections += 1
                total_conf += conf
                class_counter[turkce_etiket] += 1

        elapsed = time.perf_counter() - start
        if elapsed > 0:
            fps_values.append(1.0 / elapsed)

        frame_count += 1

    cap.release()
    avg_fps = np.mean(fps_values) if fps_values else 0
    avg_conf = total_conf / total_detections if total_detections > 0 else 0

    return {
        'frames': frame_count,
        'total_detections': total_detections,
        'avg_conf': avg_conf,
        'avg_fps': avg_fps,
        'class_counts': dict(class_counter)
    }

def plot_results(results):
    models = list(results.keys())
    total_det = [res['total_detections'] for res in results.values()]
    fps_vals = [res['avg_fps'] for res in results.values()]
    conf_vals = [res['avg_conf'] for res in results.values()]

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    bars = plt.bar(models, total_det, color='steelblue')
    plt.title('Toplam Tespit (LK Tracker ile, Maskeli)')
    plt.ylabel('Adet')
    plt.xticks(rotation=15)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)),
                 ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 2, 2)
    plt.bar(models, fps_vals, color='seagreen')
    plt.title('Ortalama FPS (LK ile)')
    plt.ylabel('FPS')
    plt.xticks(rotation=15)

    plt.subplot(2, 2, 3)
    plt.bar(models, conf_vals, color='darkorange')
    plt.title('Ortalama Güven Skoru')
    plt.ylabel('Confidence')
    plt.xticks(rotation=15)

    plt.subplot(2, 2, 4)
    all_classes = sorted(set().union(*[res['class_counts'].keys() for res in results.values()]))
    bottom = np.zeros(len(models))
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
    for i, cls in enumerate(all_classes):
        counts = [res['class_counts'].get(cls, 0) for res in results.values()]
        plt.bar(models, counts, bottom=bottom, label=cls,
                color=colors[i % len(colors)])
        bottom += np.array(counts)
    plt.title('Sınıf Bazında Dağılım (LK ile)')
    plt.ylabel('Adet')
    plt.xticks(rotation=15)
    plt.legend()

    plt.tight_layout()
    plt.savefig('benchmark_LK.png', dpi=150)
    plt.close()
    print("\n✅ Görsel rapor 'benchmark_LK.png' olarak kaydedildi.")

def main():
    video_path = "trafikvideo.mp4"
    max_frames = 300

    models = {
        "Custom (best.pt)": YoloCarDetector("best.pt"),
        "YOLOv8n": YoloCarDetector("yolov8n.pt"),
        "YOLOv8s": YoloCarDetector("yolov8s.pt"),
        "YOLO11n": YoloCarDetector("yolo11n.pt"),
        "YOLO11s": YoloCarDetector("yolo11s.pt"),
    }

    results = {}
    for name, det in models.items():
        print(f"\n--- {name} test ediliyor (LK tracker, maskeli) ---")
        res = benchmark_model_with_LK(det, video_path, max_frames=max_frames)
        results[name] = res
        print(f"  Frame: {res['frames']}, Ham tespit: {res['total_detections']}, "
              f"Ort. Güven: {res['avg_conf']:.3f}, Ort. FPS: {res['avg_fps']:.1f}")

    print("\n" + "=" * 100)
    print("KARŞILAŞTIRMA (LK Tracker, Maskeli)")
    print("=" * 100)
    header = (f"{'Model':<20} {'Frame':<8} {'Toplam':<8} {'Araba':<8} {'Motor':<8} "
              f"{'Otobüs':<8} {'Kamyonet':<10} {'Ort.Güven':<10} {'FPS':<8}")
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        cc = res['class_counts']
        print(f"{name:<20} {res['frames']:<8} {res['total_detections']:<8} "
              f"{cc.get('Araba', 0):<8} {cc.get('Motor', 0):<8} "
              f"{cc.get('Otobus', 0):<8} {cc.get('Kamyonet', 0):<10} "
              f"{res['avg_conf']:<10.3f} {res['avg_fps']:<8.1f}")

    print("\n--- Detaylı Sınıf Dağılımı ---")
    for name, res in results.items():
        print(f"\n{name}:")
        if not res['class_counts']:
            print("   Hiç hedef tespit yok")
            continue
        for cls, count in sorted(res['class_counts'].items()):
            print(f"   {cls}: {count} adet")

    plot_results(results)

if __name__ == "__main__":
    main()