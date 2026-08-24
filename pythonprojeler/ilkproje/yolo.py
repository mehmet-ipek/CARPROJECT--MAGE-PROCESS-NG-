import cv2
from ultralytics import YOLO
# -------------------------------
# MODEL YÜKLEME
# -------------------------------
def model_yukle(model_adi="yolo11n.pt"):
    """
    YOLO modelini yükler.
    yolo11n:
    - hızlı
    - webcam için uygun
    - düşük sistem kullanımı

    Daha güçlü modeller:
    yolo11s.pt
    yolo11m.pt
    yolo11l.pt
    """
    model = YOLO(model_adi)
    return model
# -------------------------------
# TEK GÖRÜNTÜDE TESPİT
# -------------------------------

def goruntu_tespit(model, frame):

    """
    Bir görüntü üzerinde nesne tespiti yapar.
    """
    sonuc = model(frame)
    return sonuc
# -------------------------------
# KAMERA İLE GERÇEK ZAMANLI TESPİT
# -------------------------------

def webcam_baslat():
    model = model_yukle()
    kamera = cv2.VideoCapture(0)
    if not kamera.isOpened():
        raise Exception("Kamera açılamadı")
    while True:
        basarili, frame = kamera.read()
        if not basarili:
            break
        # YOLO çalıştır
        sonuclar = model(frame)
        # Kutuları çiz
        for sonuc in sonuclar:
            kutular = sonuc.boxes
            for kutu in kutular:
                # koordinatlar
                x1, y1, x2, y2 = (
                    kutu.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )
                # güven oranı
                guven = float(
                    kutu.conf[0]
                )
                # sınıf numarası
                sinif_id = int(
                    kutu.cls[0]
                )
                # sınıf ismi
                isim = model.names[sinif_id]
                # kare çiz
                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )
                text = (
                    f"{isim} %{guven*100:.1f}"
                )
                cv2.putText(
                    frame,
                    text,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )
        cv2.imshow(
            "YOLO Webcam",
            frame
        )
        tus = cv2.waitKey(1)
        if tus == 27 or tus == ord("q"):
            break
    kamera.release()
    cv2.destroyAllWindows()
# -------------------------------
# VIDEO DOSYASI TESTİ
# -------------------------------
def video_test(video_yolu):

    model = model_yukle()

    model.predict(
        source=video_yolu,
        show=True
    )
# -------------------------------
# RESİM TESTİ
# -------------------------------

def resim_test(resim_yolu):

    model = model_yukle()

    sonuc = model.predict(
        source=resim_yolu,
        show=True
    )

    return sonuc
# -------------------------------
# MODEL BİLGİLERİ
# -------------------------------

def model_bilgisi():

    model = model_yukle()

    print(model.info())
# -------------------------------
# ANA PROGRAM
# -------------------------------

if __name__ == "__main__":

    print(
        """
        YOLO TEST MENÜSÜ

        1 - Webcam nesne algılama
        2 - Model bilgisi
        """
    )
    secim = input(
        "Seçim: "
    )
    if secim == "1":

        webcam_baslat()
    elif secim == "2":

        model_bilgisi()
    else:

        print(
            "Geçersiz seçim"
        )