# Imports
from ultralytics import YOLO
from paddleocr import PaddleOCR
import easyocr
import numpy as np
import cv2
import pytesseract # Tried didn't succeed
import imutils

# Setup model
reader = easyocr.Reader(['en'], gpu = False)
yoloModel = YOLO('./models/yolov8n.pt')
licensePlateModel = YOLO('./models/license_plate_detector.pt')

# Attempted fine-tuning
# licensePlateModel.train(data = "/Users/houchichan/Desktop/Dev/CS585/License-Plate-Recognition/data.yaml", epochs = 5, batch = 16)
Ocr = PaddleOCR(use_angle_cls = True, lang = 'en') # need to run only once to download and load model into memory

# Process characters that shouldn't be allowed
licenseMap = {"@": "0",
              "!": "1",
              ";": "j",
              "\'": "",
              "?": "2",
              "%": "2",
              ".": "",
              "#": "4",
              "_": "D",
              "~": "",
              "]": "T",
              "[": "C",
              "(": "C",
              ")": "I",
              "=": "-",
              "€": "E",
              "|": "",
              "\"": ""}

def formatLicensePlate(text):
    res = ""
    for t in text:
        if t in licenseMap.keys(): 
            res += licenseMap[t]
        else:
            res += t

    return res

def readLicenseImage(license_plate_crop):
    '''
    Read the license plate from the cropped image. This is done using the EasyOCR library.
    '''
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        _, text, score = detection

        text = text.upper().replace(' ', '')

        return formatLicensePlate(text), score

    return None, 0

def associate(licensePlate, tracks):
    '''
    Associate the license plate with the vehicle track given from deepsort
    '''
    x1, y1, x2, y2, _, _ = licensePlate

    for trackObj in tracks:
        carX1, carY1, carX2, carY2 = trackObj.to_tlbr()
        carId = trackObj.track_id
        if x1 > carX1 and y1 > carY1 and x2 < carX2 and y2 < carY2:
            return carX1, carY1, carX2, carY2, carId

    return -1, -1, -1, -1, -1

def deskew(image):
    '''
    Image warping to deskew the license plate
    '''
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
        borderMode = cv2.BORDER_REPLICATE)
    return rotated

def order_points(pts):
    '''
    Transpose Coordinate to Object center and find angles of the corner in a sorted format
    '''
    center = np.mean(pts)
    shifted = pts - center
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])
    ind = np.argsort(theta)
    return pts[ind]

def warpImage(img, original):  
    '''
    Warp image by finding the biggest contour and mapping it to the original image size
    '''
    biggest = np.array([])
    maxArea = 0
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index = None
    for i, cnt in enumerate(contours):  
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # To determine if the biggest Contour represents a rectangle / square
        if area > maxArea and len(approx) == 4:
            biggest = approx
            maxArea = area
            index = i  

    warped = None 
    if index is not None: 
        src = np.squeeze(biggest).astype(np.float32) 
        height = original.shape[0]
        width = original.shape[1]
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        biggest = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform and warp image
        M = cv2.getPerspectiveTransform(biggest, dst)
        img_shape = (width, height)
        warped = cv2.warpPerspective(original, M, img_shape, flags=cv2.INTER_LINEAR)

    return warped 

def renderImageEasyOCR(imagePath, imageName, outputPath):
    '''
    Render the image using the EasyOCR library
    '''
    # READING IMAGE METHOD
    frame = cv2.imread(imagePath + "/" + imageName)
    frame = cv2.resize(frame, None, fx = 1.1, fy = 1.1, interpolation = cv2.INTER_CUBIC)
    detections = licensePlateModel.predict(frame)
    sub50, temp50, temp80 = 0, 0, 0

    # Bounding box
    # we apply filtering, denoising, and edge detection to the cropped image, then apply contour detection to find the license plate
    # we then warp the image to straighten the license plate
    # we apply thresholding to the warped image to get the final image
    # we then use the EasyOCR library to read the license plate
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
                imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
                # imgBlur = cv2.bilateralFilter(imgGray, 5, 75, 75)
                # imgBlur = cv2.medianBlur(imgGray, 3)
                imgCanny = cv2.Canny(imgBlur, 0, 500, apertureSize=5)
                imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
                imgErode = cv2.erode(imgDial, kernel, iterations=2)
                warped = warpImage(imgErode, crop)

                # Final processing
                _, originalThres = cv2.threshold(imgGray, 64, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                grayWarped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, warpedThres = cv2.threshold(grayWarped, 64, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Text Reading | Based on whichever grants the best result
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

                # Performance Counter
                if score >= 0.8: 
                    temp80 += 1
                elif score >= 0.5:
                    temp50 += 1
                else:
                    sub50 += 1
            except:
                x1, y1, x2, y2, score, _ = detection
                crop = frame[int(y1): int(y2), int(x1): int(x2)]
                imgGray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, originalThres = cv2.threshold(imgGray, 64, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                text, score = readLicenseImage(originalThres)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0, 255), 4)
                cv2.putText(frame,
                            text,
                            (int(x1), int(y1 - 25 + (10 / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0, 255),
                            2)

                # Performance Counter
                if score >= 0.8: 
                    temp80 += 1
                elif score >= 0.5:
                    temp50 += 1
                else:
                    sub50 += 1
        
        print(text)
        cv2.imshow("Final", frame)
        cv2.waitKey(0)
    except:
        print("Error, Something went Wrong")
    return sub50, temp50, temp80

# TRY PADDLEOCR | DOESN'T WORK RAN FOR 10 + HOURS ON SINGULAR IMAGE WITHOUT ANY RESULT 
def renderImagePaddleOCR(imagePath, imageName, outputPath):
    '''
    Render the image using the PaddleOCR library
    '''
    img_path = imagePath + "/" + imageName
    result = Ocr.ocr(img_path, cls=False, det=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)


# TRY MORE FILTRATION / POST PROCESSING | TESSERACT KINDA BROKEN
def renderImageTesseractOCR(imagePath, imageName, outputPath):
    '''
    Render the image using the Tesseract OCR library
    '''
    original = cv2.imread(imagePath + "/" + imageName)
    detections = licensePlateModel.predict(imagePath + "/" + imageName)
    try:
        for detection in detections[0].boxes.data.tolist():
            x1, y1, x2, y2, _, _ = detection
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