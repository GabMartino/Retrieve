

from pathlib import Path

import numpy as np
from keras.api.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot as plt
from tqdm import tqdm
import tensorflow as tf


class DNNFeatureExtractor:

    def __init__(self, model, input_size):
        self.model = model
        self.input_size = input_size

    def extractFeaturesFromImage(self, imgArray, Normalize=True):

        x = np.expand_dims(imgArray, axis=0)
        x = preprocess_input(x)
        ##Extract the features
        features = self.model.predict(x)
        features = features.flatten()  ##Flatten the output
        if Normalize:
            features = features / np.linalg.norm(features)
        return features
    '''
        This method receive a list of strings. Each string is a filepath of the image
    '''
    def extractFeaturesFromDataPathList(self, fileDataPathList = "", dirPath = "", normalizeFeatures=True, limit = None):

        features_list = []
        if limit != None:
            fileDataPathList = fileDataPathList[:limit]
        for image_path in tqdm(fileDataPathList):

            img_features = self._extractFeatures(image_path)##ExtractFeatures
            if normalizeFeatures:
                img_features = img_features / np.linalg.norm(img_features)

            (className, filename) = self._getClassAndFileName(image_path, prefix=dirPath)
            image_tuple = (className, filename, img_features) ## Create a tuple of (imagepath, features)
            features_list.append(image_tuple)## insert in a list

        return features_list

    def _preprocessInput(self, img_path):
        img = image.load_img(img_path, target_size=self.input_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return x
    def _extractFeatures(self, img_path):
        ##Preprocess the image into the target size for the cnn
        img = image.load_img(img_path, target_size=self.input_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        ##Extract the features
        features = self.model.predict(x)
        features = features.flatten()  ##Flatten the output
        return features

    def _getClassAndFileName(self, imagepath, prefix):
        ##imagepath in the format "../path/path2/classname/file.jpg"

        if imagepath.startswith(prefix):
            try:
                imagepath = imagepath[len(prefix):]
                className = (imagepath.split("/",1))[0]
                fileName = (imagepath.split("/",1))[1]
                return (className, fileName)
            except:
                fileName = (imagepath.split("/", 1))[0]
                return (None , fileName)
        return None



if __name__ == "__main__":

    f = DNNFeatureExtractor( None,input_size=(224,224))

    image = f._preprocessInput("../sketches/png/airplane/1.png")
    print(type(image))
    plt.imshow(image.squeeze())
    plt.axis("off")
    plt.show()

    img = image.load_img("../sketches/png/airplane/1.png", target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    plt.imshow(x.squeeze())
    plt.axis("off")
    plt.show()