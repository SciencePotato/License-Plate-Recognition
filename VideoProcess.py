from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from util import *
import cv2

yoloModel = YOLO('./models/yolov8n.pt')
licensePlateModel = YOLO('./models/license_plate_detector.pt')

cap = cv2.VideoCapture('./data/videos/demoF.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/output/demoF.mp4', fourcc, fps, (width, height))
# DeepSort Tracker
deepSortTracker = DeepSort(max_age = 20)
carId = [2, 3, 5, 7]
association = {}
frames = -1

# Video Process
while True:
    frames += 1
    ret, frame = cap.read()
    if ret:
        detections = yoloModel(frame)[0]
        cars = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, id = detection 
            if int(id) in carId:
                cars.append([[x1, y1, x2 - x1, y2 - y1], score, id])

        # Object Track
        tracks = deepSortTracker.update_tracks(cars, frame = frame)

        licensePlate = licensePlateModel(frame)[0]
        for license in licensePlate.boxes.data.tolist():
            x1, y1, x2, y2, score, id = license
            carX1, carY1, carX2, carY2, car_Id = associate(license, tracks)

            if car_Id != -1:
                cropLicense = frame[int(y1): int(y2), int(x1): int(x2), :]
                grayCrop = cv2.cvtColor(cropLicense, cv2.COLOR_BGR2GRAY)
                _, cropThresh = cv2.threshold(grayCrop, 64, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Detectable Text over a threshold -> Render into CSV result
                text, textScore = readLicenseImage(cropThresh)
                if text != None :
                    if car_Id not in association.keys():
                        association[car_Id] = dict()

                    if text not in association[car_Id]: 
                        association[car_Id][text] = 0

                    association[car_Id][text] += 1

                text, count = "", 0
                if car_Id in association.keys():
                    for val in association[car_Id]:
                        if association[car_Id][val] > count:
                            count = association[car_Id][val]
                            text = val

                cv2.rectangle(frame, (int(carX1), int(carY1)), (int(carX2), int(carY2)), (0, 0, 255), 12)
                cv2.putText(frame,
                            text,
                            (int(int(carX1 + carX2) / 2), int(int(carY1 + carY2 + 10) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

        out.write(frame)

    else:
        # Video End / Cv2.VideoRead Error
        break

out.release()
cap.release()
