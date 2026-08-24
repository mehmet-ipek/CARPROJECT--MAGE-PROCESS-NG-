from ultralytics import YOLO

print("1. Araç modeli dönüştürülüyor...")
model_vehicle = YOLO("yolov8n.pt")
model_vehicle.export(format="openvino")

print("2. Plaka modeli dönüştürülüyor...")
model_plate = YOLO("license_plate_detector.pt")
model_plate.export(format="openvino")

print("Dönüştürme tamamlandı!")