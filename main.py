from ultralytics import YOLO
import cv2
import os

result = {}
yoloModel = YOLO('./models/yolov8n.pt')
licensePlate = YOLO('./models/license_plate_detector.pt')
vehiclesId = [2, 3, 5, 7]

# READING IMAGE METHOD
for file in os.listdir(os.fsencode("./data/images/")):
    filename = os.fsdecode(file)
    filename = "./data/images/" + filename
    detections = licensePlate.predict(filename)
    validList = []

    # Bounding box
    x1, y1, x2, y2 = 0, 0, 0, 0

# READING VIDEO