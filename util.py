from ultralytics import YOLO
import numpy as np
import string
import easyocr
import cv2

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


def renderImage(imagePath, imageName, outputPath):
    # READING IMAGE METHOD
    frame = cv2.imread(imagePath + "/" + imageName)
    detections = licensePlateModel.predict(imagePath + "/" + imageName)

    # Bounding box
    try:
        for detection in detections[0].boxes.data.tolist():
            x1, y1, x2, y2, score, id = detection
            crop = frame[int(y1): int(y2), int(x1): int(x2)]
            
            kernel = np.ones((3,3))
            imgGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
            cv2.imshow("blur", imgBlur)
            cv2.waitKey(0) 
            imgCanny = cv2.Canny(imgBlur, 0, 500, apertureSize=5)
            _, cropThresh = cv2.threshold(imgGray, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow("blur", cropThresh)
            cv2.waitKey(0) 
            cv2.imshow("canny", imgCanny)
            cv2.waitKey(0) 
            imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
            cv2.imshow("dial", imgDial)
            cv2.waitKey(0) 
            imgThres = cv2.erode(imgDial, kernel, iterations=2)
            cv2.imshow("erode", imgThres)
            cv2.waitKey(0) 
            biggest, imgContour, warped = getContours(imgThres, crop)
            cv2.imshow("contour", imgContour)
            cv2.waitKey(0) 
            cv2.imshow("Output", warped)
            cv2.waitKey(0) 
    except:
        print("Error, Something went Wrong")
    # _, cropThresh = cv2.threshold(crop, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plateText, plateScore = readLicenseImage(cropThresh)
    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0, 255), 4)
    # print(plateText, plateScore)
    # cv2.putText(frame,
    #             plateText,
    #             (int(x1), int(y1 - 25 + (10 / 2))),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.6,
    #             (0, 0, 0, 255),
    #             2)
