
from keras.models import Model
from keras.preprocessing import image
from keras.applications import vgg16
import cv2 #pip install opencv-python
import random
import sys
import argparse
import os
import random
import numpy as np
import threading


label = ''
frame = None


class Thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global label
		# Load the VGG16 network
		print("[INFO] loading network...")
		self.model = vgg16.VGG16(weights="imagenet")

		while (~(frame is None)):
			(inID, label, score) = self.predict(frame)[0]

	def predict(self, frame):
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
		img = img.reshape((1,) + img.shape)

		img = vgg16.preprocess_input(img)
		preds = self.model.predict(img)
		return vgg16.decode_predictions(preds)[0]


cap = cv2.VideoCapture(0)
if (cap.isOpened()):
	print("Camera OK")
else:
	cap.open()

keras_thread = Thread()
keras_thread.start()

while (True):
	ret, original = cap.read()

	frame = cv2.resize(original, (224, 224))

	cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow("Classification", original)

	if (cv2.waitKey(1) & 0xFF == ord('q')):
		break;


cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()