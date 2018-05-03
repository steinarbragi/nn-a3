
# dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
# blogpost: https://chunml.github.io/ChunML.github.io/project/Real-Time-Object-Recognition-part-one/
# example: https://github.com/ChunML/DeepLearning.git

import argparse
import os
import random
import numpy as np
from keras.models import Model
from keras.preprocessing import image
import keras.applications.resnet50 as res
import cv2 #pip install opencv-python

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True)
args = vars(ap.parse_args())

files = [os.path.join(args["folder"], f) for f in os.listdir(args["folder"])]
random.shuffle(files)


# Load the VGG16 network
print("[INFO] loading network...")
model = res.ResNet50(weights="imagenet")

for file in files:
	# Load the image using OpenCV
	orig = cv2.imread(file)

	# Load the image using Keras helper ultility
	print("[INFO] loading and preprocessing image...")
	img = image.load_img(file, target_size=(224, 224))
	img = image.img_to_array(img)

	# Convert (3, 224, 224) to (1, 3, 224, 224)
	# Here "1" is the number of images passed to network
	# We need it for passing batch containing serveral images in real project
	img = np.expand_dims(img, axis=0)
	img = res.preprocess_input(img)


	# Classify the image
	print("[INFO] classifying image...")
	preds = model.predict(img)
	(inID, label,score) = res.decode_predictions(preds)[0][0]

	# Display the predictions
	print("ImageNet ID: {}, Label: {}".format(inID, label))
	cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow("Classification", orig)
	cv2.waitKey(0)
