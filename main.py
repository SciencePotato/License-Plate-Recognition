from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from util import associate 

import cv2
import easyocr
import os
import numpy as np

yoloModel = YOLO('./models/yolov8n.pt')
licensePlateModel = YOLO('./models/license_plate_detector.pt')

cap = cv2.VideoCapture('./data/videos/demo.mp4')

output = {}
deepSortTracker = DeepSort(max_age = 20)
carId = [2, 3, 5, 7]

frames = -1
while True:
    frames += 1
    ret, frame = cap.read()
    if ret:
        output[frames] = {}
        detections = yoloModel(frame)[0]

        cars = []
        licensePlates = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, id = detection
            # If Identification object ID is a car 
            if (int(id)) in carId:
                # tuples of ( [left,top,w,h], confidence, detection_class )
                cars.append(([x1, y1, x2 - x1, y2 - y1], score, id))
                carCrop = frame[int(y1): int(y2), int(x1): int(x2), :]
                licensePlate = licensePlateModel(carCrop)
                for plate in licensePlate[0].boxes.data.tolist():
                    x1, y1, x2, y2, score, id = plate
                    licensePlates.append([x1, y1, x2, y2, score])
        
        # Object Track
        tracks = deepSortTracker.update_tracks(cars, frame = frame)

        for licensePlate in licensePlates:
            x1, y1, x2, y2, id = associate(licensePlate, tracks)

            if id != -1:
                cropLicense = frame[int(y1): int(y2), int(x1): int(x2), :]
                grayCrop = cv2.cvtColor(cropLicense, cv2.COLOR_BGR2GRAY)
                cropThresh = cv2.adaptiveThreshold(grayCrop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    else:
        # Video End / Cv2.VideoRead Error
        break