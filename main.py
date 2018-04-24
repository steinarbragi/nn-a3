
# dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/

# blogpost: https://chunml.github.io/ChunML.github.io/project/Real-Time-Object-Recognition-part-one/

# example: https://github.com/ChunML/DeepLearning.git

import argparse
import os
import random
import numpy as np
import cv2
from keras.models import Model
from keras.applications import applications
from keras.preprocessing import image as image_utils
from keras.applications.vgg16 import preprocess_input


model = applications.vgg16.VGG16(weights="imagenet")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True)
    args = vars(ap.parse_args())
    files = [os.path.join(args["folder"], f) for f in os.listdir(args["folder"])]
    random.shuffle(files)

    for file in files:
        #image = image_utils.load_img(file, target_size=(224, 224))
        #image = image_utils.img_to_array(image)
        image = cv2.imread(file)
        image = image.transpose((2, 0, 1))
        image = preprocess_input(image)


if __name__ == '__main__':
    main()