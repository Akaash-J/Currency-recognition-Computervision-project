from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Train the model
model.train(data="C:/Users/HP/Documents/ML_Learning/Currency detection.v2i.yolov8/data.yaml", epochs=50, imgsz=640)

