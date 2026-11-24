# License-Plate-Recognition

## Annotation conversion

Run `python3 create.py` from this directory to convert the two annotation CSVs
into YOLO class-0 labels: normalized center x/y, width, and height (not corners).
All boxes for each image are kept, and names such as `vehicle.front.jpg` produce
`vehicle.front.txt`. Non-finite values, nonpositive image dimensions, and boxes
outside the image fail before any output files are opened.

`python3 -m unittest -v test_annotations` uses synthetic CSVs and mocked file
writes. It needs only the standard library and never loads models or media.

This project is a simple license plate recognition system using OpenCV and EasyOCR. The system is able to detect and recognize license plates from images and videos.


Install dependencies from the requirements.txt file using pip.

To run detection on image dataset, run the ImageProcess.py file.


To run detection on video, run the VideoProcess.py file.


create.py is used to create an annotation file for each image in the dataset. The annotation file contains the coordinates of the license plate in the image. This is used for fine-tuning.

model.ipynb contains the code for our initial attempt at creating a custom model for license plate detection. The model was not able to detect license plates accurately, so we used a pre-trained model instead.

The data directory contains the dataset used for training and testing the model. The dataset contains images of cars with license plates. 

The model directory contains the pre-trained model used for license plate detection. The model is based on the YOLOv8 architecture.

