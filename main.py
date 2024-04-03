from ultralytics import YOLO
import cv2
import easyocr
import os
from pprint import pprint

reader = easyocr.Reader(['en'], gpu = False)
def readPlate(plate): 
    detections = reader.readtext(plate)
    
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        
        return text, score

result = []
error = []
yoloModel = YOLO('./models/yolov8n.pt')
licensePlate = YOLO('./models/best.pt')

# READING IMAGE METHOD
for file in os.listdir(os.fsencode("./data/images/")):
    filename = os.fsdecode(file)
    filename = "./data/images/" + filename
    detections = licensePlate.predict(filename)
    image = cv2.imread(filename, 0)
    validList = []

    # Bounding box
    for plate in detections:
        try:
            nparray = plate.numpy()[0]
            x1, y1, x2, y2, score, id = nparray
            crop = image[int(y1): int(y2), int(x1): int(x2)]
            _, crop_thresh = cv2.threshold(crop, 64, 255, cv2.THRESH_BINARY)
            plateText, plateScore = readPlate(crop_thresh)
            result.append([filename, plateText])
            print("License Plate Text: " + plateText + " " + filename)
        except:
            error.append(filename)
            print("Error with the image " + filename)


print(error)
print(result)
print(len(error) + " " + len(result))
# READING VIDEO