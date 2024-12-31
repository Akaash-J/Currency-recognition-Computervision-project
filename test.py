from ultralytics import YOLO

# Load the best model
model = YOLO("runs/detect/train3/weights/best.pt")

# Predict on new images
results = model.predict(source="C:/Users/HP/Documents/ML_Learning/Currency detection.v2i.yolov8/test/images", save=True)
print(results)
