import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speaking speed

# Load the trained model
model = YOLO("runs/detect/train3/weights/best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Variables for detection tracking
last_announcement_time = 0
announcement_delay = 2  # Minimum delay between announcements (in seconds)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model.predict(frame, conf=0.5, iou=0.5)

    # Annotate the frame with predictions
    annotated_frame = results[0].plot()

    # Extract detected objects
    detected_objects = results[0].boxes  # Access bounding boxes and classes
    
    # Get the current time
    current_time = time.time()

    # Process each detected object
    for obj in detected_objects:
        class_id = int(obj.cls[0])  # Class ID of the detected object
        confidence = obj.conf[0]    # Confidence score
        currency_name = model.names[class_id]  # Get currency name using class ID

        # Extract numeric part of currency name (e.g., "5-200 rupees" -> "200 rupees")
        currency_value = currency_name.split('-')[-1].strip()

        # Announce the detected currency if sufficient time has passed
        if confidence > 0.8 and current_time - last_announcement_time > announcement_delay:
            tts_engine.say(f"Detected {currency_value}")
            tts_engine.runAndWait()
            last_announcement_time = current_time

    # Display the annotated frame
    cv2.imshow("Real-Time Detection", annotated_frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
