from ultralytics import YOLO
import numpy as np
import string
import easyocr
import cv2
import os

# Setup model
reader = easyocr.Reader(['en'], gpu = False)
yoloModel = YOLO('./models/yolov8n.pt')
licensePlateModel = YOLO('./models/license_plate_detector.pt')

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, 0

def readLicenseImage(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        return text, score

    return None, 0

def associate(licensePlate, tracks):
    x1, y1, x2, y2, score, id = licensePlate

    for trackObj in tracks:
        carX1, carY1, carX2, carY2 = trackObj.to_tlbr()
        carId = trackObj.track_id
        if x1 > carX1 and y1 > carY1 and x2 < carX2 and y2 < carY2:
            return carX1, carY1, carX2, carY2, carId

    return -1, -1, -1, -1, -1

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    imgContour = img.copy()  # Change - make a copy of the image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    if index is not None: # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

    return biggest, imgContour  # Change - also return drawn image

def renderImage(imagePath, imageName, outputPath):
    # READING IMAGE METHOD
    frame = cv2.imread(imagePath + "/" + imageName, 0)
    detections = licensePlateModel.predict(imagePath + "/" + imageName)
    absPath = os.path.abspath(os.path.dirname(__file__)) + outputPath + "/" + imageName

    # Bounding box
    for plate in detections[0].boxes.data.tolist():
        try:
            x1, y1, x2, y2, score, id = plate
            crop = frame[int(y1): int(y2), int(x1): int(x2)]
            _, cropThresh = cv2.threshold(crop, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plateText, plateScore = readLicenseImage(cropThresh)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0, 255), 4)
            print(plateText, plateScore)
            cv2.putText(frame,
                        plateText,
                        (int(x1), int(y1 - 25 + (10 / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0, 255),
                        2)
        except:
            print("Error, Something went Wrong")

    cv2.imshow("Output", frame)
    cv2.waitKey(0) 
    cv2.imwrite(absPath, frame)
