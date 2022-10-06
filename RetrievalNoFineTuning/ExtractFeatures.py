import os
import pickle
import numpy as np
import tensorflow as tf
from keras import Model
from keras.applications.vgg16 import layers
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from API.DNNFeatureExtractor import DNNFeatureExtractor

'''
    This script extract features from a images using a file with all the path of the single images as entry point.
    The model is already set.
    
    Paramenter changeble:
        -> Models: VGG16, VGG19
        -> Cutting point on the CNN
        -> last pooling: AVG, MAX
        
'''
'''
    Parameters Set:
    ->model = VGG16
    -> cutting point = last layer without FC
    -> last pooling : no pooling

'''


def extractFeatures(image_paths=None, dirPath=None, limit = None, model = None):
    #dirPath = "../sketches/png/"
    #dirPath = "../mirflickr25k/mirflickr/"
    if image_paths == None:
        image_paths = []

        with open(dirPath + "filelist.txt") as f:
            for line in f.readlines():
                image_paths.append(dirPath + line.replace("\n", ""))



    input_size = (224, 224)

    '''
        CREATE AN INSTANCE OF THE FEATURE EXTRACTOR
    '''
    featureExtractor = DNNFeatureExtractor(model, input_size)

    '''
        EXTRACT THE FEATURES 
    '''
    DS_features = featureExtractor.extractFeaturesFromDataPathList(image_paths, dirPath=dirPath, limit=limit, normalizeFeatures=True)

    return DS_features





if __name__ == "__main__":

    SketchesDatasePath = "../sketches/png/"
    DistractorDatasetPath = "../mirflickr25k/mirflickr/"
    ## Extract features from the model
    base_model = VGG19(weights='imagenet', include_top=False)
    #new_model = tf.keras.models.load_model('modelLastTrainedVGG16')
    x = base_model.output
    x = base_model.get_layer('block5_pool').output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.summary()
    DS_features = extractFeatures(dirPath=DistractorDatasetPath, model=model)
    pickle.dump(DS_features, open("./DistractorFeatures/DistractorVGG19block5_pool.txt", 'wb'))