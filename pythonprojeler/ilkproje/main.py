import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL = "blaze_face_short_range.tflite"


def kamera_baslat():

    kamera = cv2.VideoCapture(0)

    if not kamera.isOpened():
        raise RuntimeError("Kamera acilamadi")


    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL
        ),
        min_detection_confidence=0.6
    )


    detector = vision.FaceDetector.create_from_options(options)


    while True:

        basarili, kare = kamera.read()

        if not basarili:
            break


        rgb = cv2.cvtColor(
            kare,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        sonuc = detector.detect(mp_image)


        for yuz in sonuc.detections:

            kutu = yuz.bounding_box

            x = kutu.origin_x
            y = kutu.origin_y
            w = kutu.width
            h = kutu.height


            cv2.rectangle(
                kare,
                (x, y),
                (x+w, y+h),
                (0,255,0),
                2
            )


            guven = yuz.categories[0].score

            cv2.putText(
                kare,
                f"Yuz %{int(guven*100)}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )


        cv2.imshow(
            "BlazeFace Kamera",
            kare
        )


        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break


    kamera.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    kamera_baslat()