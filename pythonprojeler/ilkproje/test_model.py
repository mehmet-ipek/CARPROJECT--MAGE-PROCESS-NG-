from ultralytics import YOLO
import cv2


# ==============================
# EĞİTİLMİŞ MODEL
# ==============================

MODEL_PATH = r"C:\Users\Rsa004\runs\detect\hizli_demo\weights\best.pt"

model = YOLO(MODEL_PATH)


print("\nMODEL SINIFLARI:")
print(model.names)


# ==============================
# WEBCAM
# ==============================

kamera = cv2.VideoCapture(0)


if not kamera.isOpened():
    raise Exception("Kamera açılamadı")


print("\nKamera başladı")
print("Çıkmak için Q bas\n")


while True:


    ret, frame = kamera.read()


    if not ret:
        break



    # YOLO TAHMİN

    sonuc = model.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )


    # kutuları çiz

    frame_cizili = sonuc[0].plot()



    # ekrana bas

    cv2.imshow(
        "Benim YOLO Modelim",
        frame_cizili
    )



    # Q ile çık

    if cv2.waitKey(1) & 0xff == ord("q"):
        break



kamera.release()
cv2.destroyAllWindows()