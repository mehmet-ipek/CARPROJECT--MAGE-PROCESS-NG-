from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="Common Objects.yolov11/data.yaml",

    epochs=3,

    imgsz=320,

    batch=16,

    workers=8,

    device="cpu",

    fraction=0.3,

    name="hizli_demo",

    plots=False,

    cache=False
)