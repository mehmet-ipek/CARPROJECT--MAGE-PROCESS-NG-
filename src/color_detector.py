# src/color_detector.py
import cv2
import numpy as np
from collections import Counter

def get_dominant_color(image, k=3):
    """Görüntüdeki en baskın rengi bulur (basit K-Means yaklaşımı)."""
    if image is None or image.size == 0:
        return "Bilinmeyen"
    try:
        # Resmi yeniden boyutlandır ve piksel listesine çevir
        pixels = cv2.resize(image, (50, 50)).reshape(-1, 3).astype(np.float32)
        # K-Means ile 1 renk bul
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant = centers[0].astype(int)
        # BGR -> RGB
        r, g, b = int(dominant[2]), int(dominant[1]), int(dominant[0])
        # Basit isimlendirme
        if r > 150 and g < 80 and b < 80: return "Kirmizi"
        elif r < 80 and g > 150 and b < 80: return "Yesil"
        elif r < 80 and g < 80 and b > 150: return "Mavi"
        elif r > 200 and g > 200 and b < 80: return "Sari"
        elif r > 200 and g > 200 and b > 200: return "Beyaz"
        elif r < 50 and g < 50 and b < 50: return "Siyah"
        elif abs(r - g) < 30 and abs(g - b) < 30 and r > 100: return "Gri"
        else: return f"({r},{g},{b})"
    except:
        return "Bilinmeyen"