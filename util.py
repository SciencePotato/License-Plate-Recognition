import easyocr

reader = easyocr.Reader(['en'], gpu = False)

def parseText(text):
    pass

def readLicensePlate(licensePlateCrop):
    detections = reader.readtext(licensePlateCrop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        text = parseText(text)   

        # Return value and Score
        return text, score
    
    # Error parsing / No license within 
    return None, 0


def associate(licensePlate, tracks):
    x1, y1, x2, y2, score = licensePlate

    print(len(tracks))
    for trackObj in tracks:
        carX1, carY1, carX2, carY2 = trackObj.to_tlbr()
        carId = trackObj.track_id
        print([carX1, carY1, carX2, carY2])
    
        if x1 > carX1 and y1 > carY1 and x2 < carX2 and y2 < carY2:
            print("Dock")
            return carX1, carY1, carX2, carY2, carId

    return -1, -1, -1, -1, -1

def draw():
    pass