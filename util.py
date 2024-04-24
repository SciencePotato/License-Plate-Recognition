from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
import easyocr
import numpy as np
import string
import cv2
from PIL import Image
import pytesseract
import argparse
import imutils

# Setup model
reader = easyocr.Reader(['en'], gpu = False)
yoloModel = YOLO('./models/yolov8n.pt')
licensePlateModel = YOLO('./models/license_plate_detector.pt')
# licensePlateModel.train(data = "/Users/houchichan/Desktop/Dev/CS585/License-Plate-Recognition/data.yaml", epochs = 5, batch = 16)
ocr = PaddleOCR(use_angle_cls = True, lang = 'en') # need to run only once to download and load model into memory

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
def deskew(image):
    co_ords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(co_ords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE)
    return rotated

def order_points(pts):
    # Step 1: Find object center
    center = np.mean(pts)
    # Step 2: Transpose Coordinate to object center
    shifted = pts - center
    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])
    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]

def getContours(img, original):  # Change - pass the original image too
    biggest = np.array([])
    maxArea = 0
    imgContour = original.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if area > maxArea and len(approx) == 4:
            biggest = approx
            maxArea = area
            index = i  # Also save index to contour

    warped = None  # Stores the warped license plate image
    if index is not None: # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32) # Source points
        height = original.shape[0]
        width = original.shape[1]
        # Destination points
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        biggest = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image
        img_shape = (width, height)
        warped = cv2.warpPerspective(original, M, img_shape, flags=cv2.INTER_LINEAR)

    return biggest, imgContour, warped  # Change - also return drawn image

# TRY PADDLEOCR | DOESN'T WORK
def renderImagePaddleOCR(imagePath, imageName, outputPath):
    img_path = imagePath + "/" + imageName
    result = ocr.ocr(img_path, cls=False, det=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

def renderImageEasyOCR(imagePath, imageName, outputPath):
    # READING IMAGE METHOD
    frame = cv2.imread(imagePath + "/" + imageName)
    frame = cv2.resize(frame, None, fx = 1.1, fy = 1.1, interpolation = cv2.INTER_CUBIC)
    detections = licensePlateModel.predict(frame)
    sub50, temp50, temp80 = 0, 0, 0

    # Bounding box
    try:
        for detection in detections[0].boxes.data.tolist():
            try: 
                x1, y1, x2, y2, score, id = detection
                crop = frame[int(y1): int(y2), int(x1): int(x2)]
                # norm_img = np.zeros((crop.shape[0], crop.shape[1]))
                # img = cv2.normalize(crop, norm_img, 0, 255, cv2.NORM_MINMAX)
                # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
                kernel = np.ones((3,3))
                imgGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
                imgBlur = cv2.bilateralFilter(imgGray, 5, 75, 75)
                # imgBlur = cv2.medianBlur(imgGray, 3)
                imgCanny = cv2.Canny(imgBlur, 0, 500, apertureSize=5)
                imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
                imgThres = cv2.erode(imgDial, kernel, iterations=2)
                biggest, imgContour, warped = getContours(imgThres, crop)

                # Final processing
                _, originalThres = cv2.threshold(imgGray, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                grayWarped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, warpedThres = cv2.threshold(grayWarped, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Text Reading
                oriText, oriScore = readLicenseImage(originalThres)
                warpText, warpScore = readLicenseImage(warpedThres)
                text, score = warpText, warpScore
                if oriScore > warpScore:
                    text, score = oriText, oriScore
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0, 255), 4)
                cv2.putText(frame,
                            text,
                            (int(x1), int(y1 - 25 + (10 / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0, 255),
                            2)
                if score >= 0.8: 
                    temp80 += 1
                elif score >= 0.5:
                    temp50 += 1
                else:
                    sub50 += 1
            except:
                x1, y1, x2, y2, score, id = detection
                crop = frame[int(y1): int(y2), int(x1): int(x2)]
                imgGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, originalThres = cv2.threshold(imgGray, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text, score = readLicenseImage(originalThres)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0, 255), 4)
                cv2.putText(frame,
                            text,
                            (int(x1), int(y1 - 25 + (10 / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0, 255),
                            2)
                if score >= 0.8: 
                    temp80 += 1
                elif score >= 0.5:
                    temp50 += 1
                else:
                    sub50 += 1
    except:
        print("Error, Something went Wrong")
    return sub50, temp50, temp80
    # cv2.imshow("Output", frame)
    # cv2.waitKey(0)

# TRY MORE FILTRATION / POST PROCESSING | TESSERACT KINDA BROKEN
def renderImageTesseractOCR(imagePath, imageName, outputPath):
    original = cv2.imread(imagePath + "/" + imageName)
    detections = licensePlateModel.predict(imagePath + "/" + imageName)
    try:
        for detection in detections[0].boxes.data.tolist():
            x1, y1, x2, y2, score, id = detection
            image = original[int(y1): int(y2), int(x1): int(x2)]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
            dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            dist = (dist * 255).astype("uint8")
            dist = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)
            cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            chars = []
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if w >= 35 and h >= 100:
                    chars.append(c)
            chars = np.vstack([chars[i] for i in range(0, len(chars))])
            hull = cv2.convexHull(chars)
            mask = np.zeros(image.shape[:2], dtype="uint8")
            mask = cv2.dilate(mask, None, iterations=2)
            final = cv2.bitwise_and(opening, opening, mask=mask)
    except:
        print("ERROR")