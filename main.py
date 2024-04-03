from ultralytics import YOLO
import cv2
import easyocr
import os

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
licensePlate = YOLO('./models/license_plate_detector.pt')

# READING IMAGE METHOD
for file in os.listdir(os.fsencode("./data/images/")):
    filename = os.fsdecode(file)
    filename = "./data/images/" + filename
    detections = licensePlate.predict(filename)
    image = cv2.imread(filename, 0)
    height, width  = image.shape
    validList = []

    # Bounding box
    for plate in detections[0].boxes.data.tolist():
        try:
            # nparray = plate.numpy()[0]
            x1, y1, x2, y2, score, id = plate
            crop = image[max(0, int(y1)): min(height, int(y2)), max(0, int(x1)): min(width, int(x2))]
            crop_thresh = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            plateText, plateScore = readPlate(crop_thresh)
            result.append([filename, plateText, plateScore])
        except:
            error.append(filename)


print(error)
print(result)
print(len(error))
print(len(result))
# READING VIDEO