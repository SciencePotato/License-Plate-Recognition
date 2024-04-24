# License-Plate-Recognition

This project is a simple license plate recognition system using OpenCV and EasyOCR. The system is able to detect and recognize license plates from images and videos.


Install dependencies from the requirements.txt file using pip.

To run detection on image dataset, run the ImageProcess.py file.


To run detection on video, run the VideoProcess.py file.


create.py is used to create an annotation file for each image in the dataset. The annotation file contains the coordinates of the license plate in the image. This is used for fine-tuning.

model.ipynb contains the code for our initial attempt at creating a custom model for license plate detection. The model was not able to detect license plates accurately, so we used a pre-trained model instead.

The data directory contains the dataset used for training and testing the model. The dataset contains images of cars with license plates. 

The model directory contains the pre-trained model used for license plate detection. The model is based on the YOLOv8 architecture.

