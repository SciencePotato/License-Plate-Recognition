# License-Plate-Recognition

This project is a simple license plate recognition system that uses OpenCV and EasyOCR to detect and recognize license plates in images and videos.

Install dependencies from `requirements.txt` using pip.

To run detection on the image dataset, run `ImageProcess.py`.

To run detection on a video, run `VideoProcess.py`.

`create.py` creates an annotation file for each image in the dataset. Each annotation file contains the coordinates of the license plate in the image and is used for fine-tuning.

`model.ipynb` contains the code for our initial attempt to create a custom license plate detection model. The model could not detect license plates accurately, so we used a pre-trained model instead.

The `data` directory contains the images of cars with license plates used to train and test the model.

The `models` directory contains the pre-trained license plate detection model, which is based on the YOLOv8 architecture.
