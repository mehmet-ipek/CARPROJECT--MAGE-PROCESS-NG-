import cv2
import numpy as np

class Camera:
    def __init__(self, video_path="trafikvideo.mp4"):
        self.video_path = video_path
        self.camera = cv2.VideoCapture(video_path)

        if not self.camera.isOpened():
            raise Exception(f"{video_path} dosyası veya kamera açılamadı.")

        gamma = 1.6
        inv_gamma = 1.0 / gamma
        self.gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        ret, frame = self.camera.read()
        if ret:
            cv2.imwrite("baslangic_frame.jpg", frame)
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def preprocess_frame(self, frame):
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        if mean_brightness < 90:
            frame = cv2.LUT(frame, self.gamma_table)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

        return frame

    def get_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.camera.read()
            if not ret:
                return None
        return self.preprocess_frame(frame)

    def release(self):
        self.camera.release()