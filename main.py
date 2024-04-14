import os
from util import renderImage, renderVideo
'''
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from util import *
import cv2

yoloModel = YOLO('./models/yolov8n.pt')
licensePlateModel = YOLO('./models/license_plate_detector.pt')

cap = cv2.VideoCapture('./data/videos/sample.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/output/out.mp4', fourcc, fps, (width, height))

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
        # for detection in detections.boxes.data.tolist():
        #     x1, y1, x2, y2, score, id = detection
        #     # If Identification object ID is a car 
        #     if (int(id)) in carId:
        #         # tuples of ( [left,top,w,h], confidence, detection_class )
        #         cars.append(([x1, y1, x2 - x1, y2 - y1], score, id))
        #         carCrop = frame[int(y1): int(y2), int(x1): int(x2), :]
        #         licensePlate = licensePlateModel(carCrop)
        #         for plate in licensePlate[0].boxes.data.tolist():
        #             x1, y1, x2, y2, score, id = plate
        #             licensePlates.append([x1, y1, x2, y2, score])
        
        # Object Track
        # tracks = deepSortTracker.update_tracks(cars, frame = frame)

        licensePlate = licensePlateModel(frame)[0]
        for license in licensePlate.boxes.data.tolist():
            x1, y1, x2, y2, score, id = license

            cropLicense = frame[int(y1): int(y2), int(x1): int(x2), :]
            grayCrop = cv2.cvtColor(cropLicense, cv2.COLOR_BGR2GRAY)
            # Might want to just do Threshold -> Faster
            _, cropThresh = cv2.threshold(grayCrop, 64, 255, cv2.THRESH_BINARY_INV)

            # Detectable Text over a threshold -> Render into CSV result
            text, textScore = read_license_plate(cropThresh)
            if text and textScore >= 0.50: 
                H, W = cropThresh.shape
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
                cv2.putText(frame,
                            text,
                            (int((x2 + x1) / 2), int(y1 - H - 125 + (10 / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)
        out.write(frame)

    else:
        # Video End / Cv2.VideoRead Error
        break

out.release()
cap.release()
'''


# PROCESSING IAMGE
# READING IMAGE METHOD
inPath = "./data/images"
output = "/data/output/images"
for file in os.listdir(os.fsencode(inPath)):
    filename = os.fsdecode(file)
    renderImage(inPath, filename, output)