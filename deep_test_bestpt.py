import cv2
import os
import time
import json
import numpy as np

from collections import Counter

from src.yolo_detector import YoloCarDetector


# ============================================================
# AYARLAR
# ============================================================

VIDEO_PATH = "trafikvideo.mp4"
MASK_PATH = "trafikvideo.jpg"
MODEL_PATH = "best.pt"

MAX_FRAMES = 300

# Confidence sweep
CONF_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50
]

RESULT_DIR = "deep_test_bestpt"
LOG_DIR = os.path.join(
    RESULT_DIR,
    "logs"
)

IMAGE_DIR = os.path.join(
    RESULT_DIR,
    "sample_frames"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)


# ============================================================
# SINIF İSİMLERİ
# ============================================================

def get_class_name(model, cls_id):

    raw_name = model.names.get(
        int(cls_id),
        None
    )

    if raw_name is None:
        return f"ID:{cls_id}"

    raw_name = str(
        raw_name
    ).lower().strip()

    mapping = {
        "car": "Araba",
        "cars": "Araba",
        "motorcycle": "Motor",
        "motorbike": "Motor",
        "bus": "Otobus",
        "truck": "Kamyonet",
        "lorry": "Kamyonet"
    }

    return mapping.get(
        raw_name,
        f"ID:{cls_id}"
    )


# ============================================================
# MASKE
# ============================================================

def load_mask(frame):

    raw = cv2.imread(
        MASK_PATH,
        cv2.IMREAD_GRAYSCALE
    )

    if raw is None:
        print(
            "[UYARI] trafikvideo.jpg bulunamadı."
        )
        return None

    h, w = frame.shape[:2]

    mask = cv2.resize(
        raw,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mask


# ============================================================
# GECE ÖN İŞLEME
# ============================================================

def preprocess_day_only(frame):

    """
    Gündüz videosu olduğu için varsayılan testte
    frame'i değiştirmiyoruz.
    """

    return frame.copy()


def preprocess_night_style(frame):

    """
    Mevcut Camera.py'deki gece işleme mantığının
    aynısı.
    """

    result = frame.copy()

    gamma = 1.6
    inv_gamma = 1.0 / gamma

    gamma_table = np.array(
        [
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]
    ).astype("uint8")

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2GRAY
    )

    mean_brightness = float(
        np.mean(gray)
    )

    if mean_brightness < 90:

        result = cv2.LUT(
            result,
            gamma_table
        )

        lab = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        cl = clahe.apply(l)

        result = cv2.cvtColor(
            cv2.merge((cl, a, b)),
            cv2.COLOR_LAB2BGR
        )

        result = cv2.GaussianBlur(
            result,
            (3, 3),
            0
        )

    return result


# ============================================================
# MASKE UYGULA
# ============================================================

def center_in_mask(
    mask,
    x,
    y,
    w,
    h
):

    if mask is None:
        return True

    cx = int(
        x + w / 2
    )

    cy = int(
        y + h / 2
    )

    mh, mw = mask.shape[:2]

    if not (
        0 <= cx < mw
        and
        0 <= cy < mh
    ):
        return False

    return mask[cy, cx] > 0


# ============================================================
# YOLO'YU DOĞRUDAN ÇALIŞTIR
# ============================================================

def raw_yolo_detect(
    model,
    frame,
    conf
):

    results = model.predict(

        source=frame,

        conf=conf,

        imgsz=640,

        max_det=500,

        agnostic_nms=False,

        verbose=False
    )

    detections = []

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    for box in result.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        cls_id = int(
            box.cls[0]
            .cpu()
            .numpy()
        )

        confidence = float(
            box.conf[0]
            .cpu()
            .numpy()
        )

        x1 = int(
            max(
                0,
                x1
            )
        )

        y1 = int(
            max(
                0,
                y1
            )
        )

        x2 = int(
            min(
                frame.shape[1] - 1,
                x2
            )
        )

        y2 = int(
            min(
                frame.shape[0] - 1,
                y2
            )
        )

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        detections.append(
            (
                x1,
                y1,
                w,
                h,
                cls_id,
                confidence
            )
        )

    return detections


# ============================================================
# İSTATİSTİK
# ============================================================

def calculate_stats(
    frame_records,
    frame_count
):

    total_detection = 0
    confidence_values = []
    class_counter = Counter()

    frames_with_detection = 0

    for record in frame_records:

        detections = record[
            "detections"
        ]

        if detections:
            frames_with_detection += 1

        for det in detections:

            cls_id = det[4]
            conf = det[5]

            total_detection += 1

            confidence_values.append(
                conf
            )

            class_counter[
                record[
                    "class_map"
                ].get(
                    cls_id,
                    f"ID:{cls_id}"
                )
            ] += 1

    avg_conf = (
        float(
            np.mean(
                confidence_values
            )
        )
        if confidence_values
        else 0.0
    )

    min_conf = (
        float(
            np.min(
                confidence_values
            )
        )
        if confidence_values
        else 0.0
    )

    max_conf = (
        float(
            np.max(
                confidence_values
            )
        )
        if confidence_values
        else 0.0
    )

    coverage = (
        frames_with_detection
        /
        frame_count
        *
        100
        if frame_count > 0
        else 0
    )

    avg_per_frame = (
        total_detection
        /
        frame_count
        if frame_count > 0
        else 0
    )

    return {
        "frames": frame_count,
        "frames_with_detection":
            frames_with_detection,
        "coverage":
            coverage,
        "total_detection":
            total_detection,
        "avg_per_frame":
            avg_per_frame,
        "avg_conf":
            avg_conf,
        "min_conf":
            min_conf,
        "max_conf":
            max_conf,
        "class_counts":
            dict(class_counter)
    }


# ============================================================
# TEK TEST
# ============================================================

def run_test(
    detector_model,
    test_name,
    conf_threshold,
    use_mask,
    preprocess_mode
):

    print()
    print("=" * 90)
    print(
        f"TEST: {test_name}"
    )
    print("=" * 90)

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Video açılamadı: "
            + VIDEO_PATH
        )

    frame_count = 0

    frame_records = []

    fps_values = []

    mask = None

    class_map = {
        int(k): get_class_name(
            detector_model,
            k
        )
        for k in detector_model.names.keys()
    }

    while frame_count < MAX_FRAMES:

        ret, original_frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        if preprocess_mode == "normal":

            frame = preprocess_day_only(
                original_frame
            )

        else:

            frame = preprocess_night_style(
                original_frame
            )

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        if mask is None:

            mask = load_mask(
                frame
            )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        start = time.perf_counter()

        raw_detections = raw_yolo_detect(
            detector_model,
            frame,
            conf_threshold
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        fps = (
            1.0 / elapsed
            if elapsed > 0
            else 0
        )

        fps_values.append(
            fps
        )

        # ----------------------------------------------------
        # MASKE
        # ----------------------------------------------------

        final_detections = []

        for det in raw_detections:

            x, y, w, h, cls_id, conf = det

            if use_mask:

                if not center_in_mask(
                    mask,
                    x,
                    y,
                    w,
                    h
                ):
                    continue

            final_detections.append(
                det
            )

        frame_records.append(
            {
                "frame":
                    frame_count,

                "detections":
                    final_detections,

                "raw_count":
                    len(raw_detections),

                "class_map":
                    class_map,

                "fps":
                    fps
            }
        )

    cap.release()

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    stats = calculate_stats(
        frame_records,
        frame_count
    )

    stats["avg_fps"] = (
        float(
            np.mean(
                fps_values
            )
        )
        if fps_values
        else 0
    )

    stats["use_mask"] = use_mask
    stats["preprocess"] = preprocess_mode
    stats["conf_threshold"] = conf_threshold

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    safe_name = (
        test_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
    )

    log_path = os.path.join(
        LOG_DIR,
        safe_name + ".txt"
    )

    with open(
        log_path,
        "w",
        encoding="utf-8"
    ) as log:

        log.write(
            f"TEST: {test_name}\n"
        )

        log.write(
            f"CONF: {conf_threshold}\n"
        )

        log.write(
            f"MASK: {use_mask}\n"
        )

        log.write(
            f"PREPROCESS: {preprocess_mode}\n"
        )

        log.write(
            "=" * 80
            + "\n"
        )

        log.write(
            f"Frame: "
            f"{stats['frames']}\n"
        )

        log.write(
            f"Tespit bulunan frame: "
            f"{stats['frames_with_detection']}\n"
        )

        log.write(
            f"Coverage: "
            f"%{stats['coverage']:.2f}\n"
        )

        log.write(
            f"Toplam detection: "
            f"{stats['total_detection']}\n"
        )

        log.write(
            f"Frame başına: "
            f"{stats['avg_per_frame']:.2f}\n"
        )

        log.write(
            f"Avg Conf: "
            f"{stats['avg_conf']:.3f}\n"
        )

        log.write(
            f"Min Conf: "
            f"{stats['min_conf']:.3f}\n"
        )

        log.write(
            f"Max Conf: "
            f"{stats['max_conf']:.3f}\n"
        )

        log.write(
            f"FPS: "
            f"{stats['avg_fps']:.2f}\n"
        )

        log.write(
            "\nSINIF DAĞILIMI\n"
        )

        for cls_name, count in sorted(
            stats["class_counts"].items()
        ):

            log.write(
                f"{cls_name}: "
                f"{count}\n"
            )

        log.write(
            "\nFRAME DETAYI\n"
        )

        for record in frame_records:

            log.write(
                f"\nFrame "
                f"{record['frame']}\n"
            )

            log.write(
                f"Raw detection: "
                f"{record['raw_count']}\n"
            )

            log.write(
                f"Final detection: "
                f"{len(record['detections'])}\n"
            )

            for det in record["detections"]:

                x, y, w, h, cls_id, conf = det

                log.write(
                    f"  "
                    f"class={record['class_map'].get(cls_id)} "
                    f"id={cls_id} "
                    f"conf={conf:.3f} "
                    f"box=({x},{y},{w},{h})\n"
                )

    return stats, frame_records


# ============================================================
# ÖRNEK FRAME GÖRSELLERİ
# ============================================================

def create_sample_images(
    detector_model
):

    print()
    print(
        "[INFO] Örnek frame görselleri oluşturuluyor..."
    )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        return

    sample_frames = [
        1,
        50,
        100,
        150,
        200,
        250,
        300
    ]

    mask = None

    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_frame = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if current_frame not in sample_frames:
            continue

        if mask is None:

            mask = load_mask(
                frame
            )

        detections = raw_yolo_detect(
            detector_model,
            frame,
            0.15
        )

        # ----------------------------------------------------
        # HAM
        # ----------------------------------------------------

        raw_img = frame.copy()

        for det in detections:

            x, y, w, h, cls_id, conf = det

            class_name = get_class_name(
                detector_model,
                cls_id
            )

            cv2.rectangle(
                raw_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                raw_img,
                f"{class_name} {conf:.2f}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        raw_path = os.path.join(
            IMAGE_DIR,
            f"frame_{current_frame:03d}_raw.jpg"
        )

        cv2.imwrite(
            raw_path,
            raw_img
        )

        # ----------------------------------------------------
        # MASKELI
        # ----------------------------------------------------

        masked_img = frame.copy()

        for det in detections:

            x, y, w, h, cls_id, conf = det

            if not center_in_mask(
                mask,
                x,
                y,
                w,
                h
            ):
                continue

            class_name = get_class_name(
                detector_model,
                cls_id
            )

            cv2.rectangle(
                masked_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                masked_img,
                f"{class_name} {conf:.2f}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        masked_path = os.path.join(
            IMAGE_DIR,
            f"frame_{current_frame:03d}_masked.jpg"
        )

        cv2.imwrite(
            masked_path,
            masked_img
        )

        # ----------------------------------------------------
        # Sadece maske
        # ----------------------------------------------------

        mask_path = os.path.join(
            IMAGE_DIR,
            f"frame_{current_frame:03d}_mask.jpg"
        )

        cv2.imwrite(
            mask_path,
            mask
        )

        saved += 1

        if saved >= len(sample_frames):
            break

    cap.release()

    print(
        f"[OK] {saved} frame örneği kaydedildi."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 90)
    print("# BEST.PT DERİN TEST")
    print("#" * 90)

    print(
        f"Video : {VIDEO_PATH}"
    )

    print(
        f"Maske : {MASK_PATH}"
    )

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Frame : {MAX_FRAMES}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    detector = YoloCarDetector(
        MODEL_PATH
    )

    model = detector.model

    # --------------------------------------------------------
    # TEST SONUÇLARI
    # --------------------------------------------------------

    all_results = {}

    # ========================================================
    # TEST 1
    # CONFIDENCE SWEEP + MASKE
    # NORMAL GÖRÜNTÜ
    # ========================================================

    for conf in CONF_THRESHOLDS:

        name = (
            f"conf_{conf:.2f}_mask_normal"
        )

        stats, _ = run_test(
            model,
            name,
            conf,
            True,
            "normal"
        )

        all_results[name] = stats

    # ========================================================
    # TEST 2
    # CONFIDENCE SWEEP + MASKESIZ
    # ========================================================

    for conf in [
        0.10,
        0.15,
        0.20,
        0.30
    ]:

        name = (
            f"conf_{conf:.2f}_nomask_normal"
        )

        stats, _ = run_test(
            model,
            name,
            conf,
            False,
            "normal"
        )

        all_results[name] = stats

    # ========================================================
    # TEST 3
    # NORMAL VS GECE PREPROCESS
    # ========================================================

    for use_mask in [
        True,
        False
    ]:

        name_normal = (
            "conf_0.15_"
            + (
                "mask"
                if use_mask
                else "nomask"
            )
            + "_normal"
        )

        name_night = (
            "conf_0.15_"
            + (
                "mask"
                if use_mask
                else "nomask"
            )
            + "_nightpreprocess"
        )

        stats_night, _ = run_test(
            model,
            name_night,
            0.15,
            use_mask,
            "night"
        )

        all_results[name_night] = (
            stats_night
        )

    # ========================================================
    # ÖRNEK GÖRSELLER
    # ========================================================

    create_sample_images(
        model
    )

    # ========================================================
    # SONUÇ ÖZETİ JSON
    # ========================================================

    json_path = os.path.join(
        RESULT_DIR,
        "results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # CONF SWEEP TABLOSU
    # ========================================================

    print()
    print()
    print("=" * 115)
    print("CONFIDENCE SWEEP - MASKELİ - NORMAL")
    print("=" * 115)

    print(
        f"{'CONF':<8}"
        f"{'Coverage':<12}"
        f"{'Detection':<12}"
        f"{'Avg/Frame':<12}"
        f"{'AvgConf':<10}"
        f"{'FPS':<10}"
    )

    print("-" * 115)

    for conf in CONF_THRESHOLDS:

        name = (
            f"conf_{conf:.2f}_mask_normal"
        )

        r = all_results[name]

        print(
            f"{conf:<8.2f}"
            f"%{r['coverage']:<11.2f}"
            f"{r['total_detection']:<12}"
            f"{r['avg_per_frame']:<12.2f}"
            f"{r['avg_conf']:<10.3f}"
            f"{r['avg_fps']:<10.2f}"
        )

    # ========================================================
    # MASK TESTİ
    # ========================================================

    print()
    print()
    print("=" * 115)
    print("MASKE ETKİSİ")
    print("=" * 115)

    for conf in [
        0.10,
        0.15,
        0.20,
        0.30
    ]:

        mask_name = (
            f"conf_{conf:.2f}_mask_normal"
        )

        nomask_name = (
            f"conf_{conf:.2f}_nomask_normal"
        )

        a = all_results[
            mask_name
        ]

        b = all_results[
            nomask_name
        ]

        difference = (
            b["total_detection"]
            -
            a["total_detection"]
        )

        print(
            f"CONF {conf:.2f} | "
            f"Maskeli={a['total_detection']} | "
            f"Masksiz={b['total_detection']} | "
            f"Fark={difference}"
        )

    # ========================================================
    # PREPROCESS TESTİ
    # ========================================================

    print()
    print()
    print("=" * 115)
    print("ÖN İŞLEME ETKİSİ")
    print("=" * 115)

    normal_name = (
        "conf_0.15_mask_normal"
    )

    night_name = (
        "conf_0.15_mask_nightpreprocess"
    )

    normal = all_results[
        normal_name
    ]

    night = all_results[
        night_name
    ]

    print(
        f"Normal       : "
        f"{normal['total_detection']}"
    )

    print(
        f"Night Style  : "
        f"{night['total_detection']}"
    )

    print(
        f"Normal Conf  : "
        f"{normal['avg_conf']:.3f}"
    )

    print(
        f"Night Conf   : "
        f"{night['avg_conf']:.3f}"
    )

    # ========================================================
    # OTOMATİK TEŞHİS
    # ========================================================

    print()
    print()
    print("#" * 90)
    print("OTOMATİK TEŞHİS")
    print("#" * 90)

    sweep_results = []

    for conf in CONF_THRESHOLDS:

        name = (
            f"conf_{conf:.2f}_mask_normal"
        )

        sweep_results.append(
            (
                conf,
                all_results[name]
            )
        )

    best_by_detection = max(
        sweep_results,
        key=lambda x:
        x[1]["total_detection"]
    )

    print()
    print(
        "En yüksek detection:"
    )

    print(
        f"CONF={best_by_detection[0]:.2f} | "
        f"Detection="
        f"{best_by_detection[1]['total_detection']}"
    )

    conf_015 = all_results[
        "conf_0.15_mask_normal"
    ]

    conf_030 = all_results[
        "conf_0.30_mask_normal"
    ]

    print()
    print(
        "0.15 → 0.30 karşılaştırması:"
    )

    print(
        f"0.15 = "
        f"{conf_015['total_detection']}"
    )

    print(
        f"0.30 = "
        f"{conf_030['total_detection']}"
    )

    ratio = (
        conf_015["total_detection"]
        /
        max(
            1,
            conf_030["total_detection"]
        )
    )

    print(
        f"Detection oranı: "
        f"{ratio:.2f}x"
    )

    mask_015 = all_results[
        "conf_0.15_mask_normal"
    ]

    nomask_015 = all_results[
        "conf_0.15_nomask_normal"
    ]

    mask_ratio = (
        mask_015["total_detection"]
        /
        max(
            1,
            nomask_015["total_detection"]
        )
    )

    print()
    print(
        "Maske retention:"
    )

    print(
        f"{mask_ratio * 100:.2f}%"
    )

    if ratio > 1.5:

        print()
        print(
            "[SONUC] Confidence eşigi önemli."
        )

        print(
            "Model düşük confidence ile "
            "çok daha fazla detection üretiyor."
        )

    else:

        print()
        print(
            "[SONUC] Confidence değişiminin "
            "etkisi sınırlı."
        )

    if mask_ratio < 0.70:

        print()
        print(
            "[UYARI] Maske çok fazla detection "
            "eliyor olabilir."
        )

    else:

        print()
        print(
            "[OK] Maske detection'ların "
            "çoğunu koruyor."
        )

    print()
    print(
        "Detaylı sonuçlar:"
    )

    print(
        RESULT_DIR
    )

    print()
    print(
        "✅ TEST TAMAMLANDI."
    )


if __name__ == "__main__":
    main()