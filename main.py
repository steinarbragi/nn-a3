
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



def main():

    model = res.ResNet50(weights='imagenet')

    img_path = 'images/test/badger.jpg'

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = res.preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', res.decode_predictions(preds))

"""
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

"""
if __name__ == '__main__':
    main()