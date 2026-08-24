import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from src.yolo_detector import YoloCarDetector


# ============================================================
# AYARLAR
# ============================================================

VIDEO_PATH = "trafikvideo.mp4"
MASK_PATH = "trafikvideo.jpg"

MAX_FRAMES = 300

LOG_DIR = "benchmark_logs"
REPORT_DIR = "benchmark_reports"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# MODEL SINIFLARINI TÜRKÇEYE ÇEVİR
# ============================================================

def get_class_name(model, cls_id):

    raw_name = model.names.get(int(cls_id), None)

    if raw_name is None:
        return f"ID:{cls_id}"

    raw_name = str(raw_name).lower().strip()

    class_map = {
        "car": "Araba",
        "cars": "Araba",
        "motorcycle": "Motor",
        "motorbike": "Motor",
        "bus": "Otobus",
        "truck": "Kamyonet",
        "lorry": "Kamyonet"
    }

    return class_map.get(
        raw_name,
        f"ID:{cls_id}"
    )


# ============================================================
# MASKE YÜKLE
# ============================================================

def load_mask(mask_path, frame_shape):

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(
            f"Maske bulunamadı: {mask_path}"
        )

    h, w = frame_shape[:2]

    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # Siyah = 0
    # Beyaz = 255
    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mask


# ============================================================
# TEK MODELİ TEST ET
# ============================================================

def benchmark_model(
    model_name,
    model_path,
    max_frames=300
):

    print("\n")
    print("=" * 90)
    print(f"TEST BAŞLADI: {model_name}")
    print("=" * 90)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    detector = YoloCarDetector(
        model_path
    )

    model = detector.model

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Video açılamadı: {VIDEO_PATH}"
        )

    # --------------------------------------------------------
    # MASKE
    # --------------------------------------------------------

    mask = None

    # --------------------------------------------------------
    # İSTATİSTİKLER
    # --------------------------------------------------------

    frame_count = 0

    frames_with_detection = 0

    total_detections = 0

    confidence_values = []

    fps_values = []

    class_counter = Counter()

    # Her frame'in detection sayısı
    frame_detection_counts = []

    # En yüksek / düşük confidence
    max_conf = 0.0
    min_conf = 1.0

    # --------------------------------------------------------
    # LOG DOSYASI
    # --------------------------------------------------------

    safe_name = (
        model_name
        .replace("(", "")
        .replace(")", "")
        .replace(".", "_")
        .replace(" ", "_")
    )

    log_path = os.path.join(
        LOG_DIR,
        f"{safe_name}.log"
    )

    with open(
        log_path,
        "w",
        encoding="utf-8"
    ) as log:

        log.write(
            f"MODEL: {model_name}\n"
        )

        log.write(
            f"MODEL DOSYASI: {model_path}\n"
        )

        log.write(
            f"VIDEO: {VIDEO_PATH}\n"
        )

        log.write(
            f"MASKE: {MASK_PATH}\n"
        )

        log.write(
            f"TEST FRAME: {max_frames}\n"
        )

        log.write(
            "\n"
            + "=" * 90
            + "\n\n"
        )

        # ====================================================
        # 300 FRAME
        # ====================================================

        while frame_count < max_frames:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # ------------------------------------------------
            # MASKEYİ İLK FRAME'DE HAZIRLA
            # ------------------------------------------------

            if mask is None:

                mask = load_mask(
                    MASK_PATH,
                    frame.shape
                )

            # ------------------------------------------------
            # FPS BAŞLANGICI
            # ------------------------------------------------

            start_time = time.perf_counter()

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            # ÖNEMLİ:
            # detector.detect() kendi içinde yine maskeyi
            # kontrol ediyor.
            #
            # Ama benchmark tarafında da maskeyi kontrol
            # ediyoruz ki test mantığı tamamen açık olsun.

            detections = detector.detect(
                frame
            )

            # ------------------------------------------------
            # FRAME DETECTIONLERİ
            # ------------------------------------------------

            valid_detections = []

            for det in detections:

                x, y, w, h, cls_id, conf = det

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                cls_id = int(cls_id)
                conf = float(conf)

                # --------------------------------------------
                # MERKEZ
                # --------------------------------------------

                cx = int(
                    x + w / 2
                )

                cy = int(
                    y + h / 2
                )

                # --------------------------------------------
                # FRAME DIŞI KONTROL
                # --------------------------------------------

                if not (
                    0 <= cx < frame.shape[1]
                    and
                    0 <= cy < frame.shape[0]
                ):
                    continue

                # --------------------------------------------
                # MASKE
                # --------------------------------------------

                if mask[cy, cx] == 0:
                    continue

                valid_detections.append(
                    (
                        x,
                        y,
                        w,
                        h,
                        cls_id,
                        conf
                    )
                )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            elapsed = (
                time.perf_counter()
                - start_time
            )

            fps = (
                1.0 / elapsed
                if elapsed > 0
                else 0
            )

            fps_values.append(
                fps
            )

            # ------------------------------------------------
            # FRAME İSTATİSTİĞİ
            # ------------------------------------------------

            detection_count = len(
                valid_detections
            )

            frame_detection_counts.append(
                detection_count
            )

            if detection_count > 0:
                frames_with_detection += 1

            # ------------------------------------------------
            # LOG
            # ------------------------------------------------

            log.write(
                f"### Frame {frame_count}\n"
            )

            log.write(
                f"Tespit sayısı: "
                f"{detection_count}\n"
            )

            for det in valid_detections:

                x, y, w, h, cls_id, conf = det

                class_name = get_class_name(
                    model,
                    cls_id
                )

                total_detections += 1

                class_counter[
                    class_name
                ] += 1

                confidence_values.append(
                    conf
                )

                max_conf = max(
                    max_conf,
                    conf
                )

                min_conf = min(
                    min_conf,
                    conf
                )

                log.write(
                    f"CLASS_ID:{cls_id} | "
                    f"CLASS:{class_name} | "
                    f"CONF:{conf:.3f} | "
                    f"BOX=({x},{y},{w},{h})\n"
                )

            if detection_count == 0:
                log.write(
                    "Tespit yok\n"
                )

            log.write(
                f"FPS: {fps:.2f}\n\n"
            )

    cap.release()

    # ========================================================
    # İSTATİSTİKLER
    # ========================================================

    avg_conf = (
        float(
            np.mean(
                confidence_values
            )
        )
        if confidence_values
        else 0.0
    )

    avg_fps = (
        float(
            np.mean(
                fps_values
            )
        )
        if fps_values
        else 0.0
    )

    frame_coverage = (
        (
            frames_with_detection
            /
            frame_count
        ) * 100
        if frame_count > 0
        else 0
    )

    avg_detections_per_frame = (
        total_detections
        /
        frame_count
        if frame_count > 0
        else 0
    )

    if not confidence_values:
        min_conf = 0.0

    # ========================================================
    # ÖZETİ LOGA EKLE
    # ========================================================

    with open(
        log_path,
        "a",
        encoding="utf-8"
    ) as log:

        log.write(
            "\n"
            + "=" * 90
            + "\n"
        )

        log.write(
            "ÖZET\n"
        )

        log.write(
            f"Frame: {frame_count}\n"
        )

        log.write(
            f"Tespit bulunan frame: "
            f"{frames_with_detection}\n"
        )

        log.write(
            f"Frame Coverage: "
            f"%{frame_coverage:.2f}\n"
        )

        log.write(
            f"Toplam detection: "
            f"{total_detections}\n"
        )

        log.write(
            f"Frame başına ortalama detection: "
            f"{avg_detections_per_frame:.2f}\n"
        )

        log.write(
            f"Ortalama confidence: "
            f"{avg_conf:.3f}\n"
        )

        log.write(
            f"Minimum confidence: "
            f"{min_conf:.3f}\n"
        )

        log.write(
            f"Maksimum confidence: "
            f"{max_conf:.3f}\n"
        )

        log.write(
            f"Ortalama FPS: "
            f"{avg_fps:.2f}\n"
        )

        log.write(
            "\nSınıf dağılımı:\n"
        )

        for cls_name, count in sorted(
            class_counter.items()
        ):

            log.write(
                f"{cls_name}: "
                f"{count}\n"
            )

    # ========================================================
    # EKRANA YAZ
    # ========================================================

    print(
        f"\n{model_name}"
    )

    print(
        f"Frame: {frame_count}"
    )

    print(
        f"Tespit bulunan frame: "
        f"{frames_with_detection}"
    )

    print(
        f"Frame Coverage: "
        f"%{frame_coverage:.2f}"
    )

    print(
        f"Toplam detection: "
        f"{total_detections}"
    )

    print(
        f"Frame başına ortalama detection: "
        f"{avg_detections_per_frame:.2f}"
    )

    print(
        f"Ortalama confidence: "
        f"{avg_conf:.3f}"
    )

    print(
        f"Ortalama FPS: "
        f"{avg_fps:.2f}"
    )

    print(
        "\nSınıf dağılımı:"
    )

    for cls_name, count in sorted(
        class_counter.items()
    ):

        print(
            f"  {cls_name}: {count}"
        )

    print(
        f"\nLog: {log_path}"
    )

    return {
        "frames": frame_count,
        "frames_with_detection":
            frames_with_detection,
        "frame_coverage":
            frame_coverage,
        "total_detections":
            total_detections,
        "avg_detections_per_frame":
            avg_detections_per_frame,
        "avg_conf":
            avg_conf,
        "min_conf":
            min_conf,
        "max_conf":
            max_conf,
        "avg_fps":
            avg_fps,
        "class_counts":
            dict(class_counter)
    }


# ============================================================
# GRAFİKLER
# ============================================================

def create_graphs(results):

    model_names = list(
        results.keys()
    )

    # --------------------------------------------------------
    # 1. TOPLAM DETECTION
    # --------------------------------------------------------

    values = [
        results[name]["total_detections"]
        for name in model_names
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        model_names,
        values
    )

    plt.title(
        "300 Frame - Toplam Tespit Sayısı"
    )

    plt.ylabel(
        "Detection Sayısı"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "toplam_detection.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # 2. FRAME COVERAGE
    # --------------------------------------------------------

    values = [
        results[name]["frame_coverage"]
        for name in model_names
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        model_names,
        values
    )

    plt.title(
        "300 Frame - Frame Coverage"
    )

    plt.ylabel(
        "Coverage (%)"
    )

    plt.xticks(
        rotation=15
    )

    plt.ylim(
        0,
        100
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "frame_coverage.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # 3. CONFIDENCE
    # --------------------------------------------------------

    values = [
        results[name]["avg_conf"]
        for name in model_names
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        model_names,
        values
    )

    plt.title(
        "Ortalama Confidence"
    )

    plt.ylabel(
        "Confidence"
    )

    plt.xticks(
        rotation=15
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "average_confidence.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # 4. FPS
    # --------------------------------------------------------

    values = [
        results[name]["avg_fps"]
        for name in model_names
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        model_names,
        values
    )

    plt.title(
        "Ortalama FPS"
    )

    plt.ylabel(
        "FPS"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "average_fps.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # 5. SINIF DAĞILIMI
    # --------------------------------------------------------

    all_classes = set()

    for result in results.values():

        all_classes.update(
            result["class_counts"].keys()
        )

    all_classes = sorted(
        all_classes
    )

    plt.figure(
        figsize=(14, 7)
    )

    x = np.arange(
        len(model_names)
    )

    width = 0.18

    for i, class_name in enumerate(
        all_classes
    ):

        values = [
            results[name]
            ["class_counts"]
            .get(
                class_name,
                0
            )
            for name in model_names
        ]

        plt.bar(
            x + (i * width),
            values,
            width=width,
            label=class_name
        )

    plt.title(
        "Sınıf Bazında Tespit Dağılımı"
    )

    plt.ylabel(
        "Detection Sayısı"
    )

    plt.xticks(
        x + width,
        model_names,
        rotation=15
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "class_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    models = {
        "Custom (best.pt)": "best.pt",
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt",
        "YOLO11n": "yolo11n.pt",
        "YOLO11s": "yolo11s.pt"
    }

    results = {}

    # --------------------------------------------------------
    # MODELLERİ TEK TEK ÇALIŞTIR
    # --------------------------------------------------------

    for model_name, model_path in models.items():

        results[
            model_name
        ] = benchmark_model(

            model_name=model_name,

            model_path=model_path,

            max_frames=MAX_FRAMES
        )

    # --------------------------------------------------------
    # KARŞILAŞTIRMA TABLOSU
    # --------------------------------------------------------

    print("\n")
    print("=" * 120)
    print("FINAL KARŞILAŞTIRMA")
    print("=" * 120)

    print(
        f"{'Model':<22}"
        f"{'Frame':<8}"
        f"{'Coverage':<12}"
        f"{'Detection':<12}"
        f"{'Avg Conf':<12}"
        f"{'FPS':<10}"
    )

    print("-" * 120)

    for model_name, result in results.items():

        print(
            f"{model_name:<22}"
            f"{result['frames']:<8}"
            f"%{result['frame_coverage']:<11.2f}"
            f"{result['total_detections']:<12}"
            f"{result['avg_conf']:<12.3f}"
            f"{result['avg_fps']:<10.2f}"
        )

    # --------------------------------------------------------
    # SINIF DAĞILIMLARI
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SINIF DAĞILIMLARI")
    print("=" * 80)

    for model_name, result in results.items():

        print(
            f"\n--- {model_name} ---"
        )

        for cls_name, count in sorted(
            result["class_counts"].items()
        ):

            print(
                f"{cls_name}: {count}"
            )

    # --------------------------------------------------------
    # GRAFİKLER
    # --------------------------------------------------------

    create_graphs(
        results
    )

    print("\n")
    print(
        "✅ Benchmark tamamlandı."
    )

    print(
        f"✅ Loglar: {LOG_DIR}"
    )

    print(
        f"✅ Grafikler: {REPORT_DIR}"
    )


if __name__ == "__main__":
    main()