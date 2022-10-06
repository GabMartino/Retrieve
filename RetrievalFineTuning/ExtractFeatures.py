import os
import pickle
from keras import Model
from keras.models import load_model
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
    DistractorDatasetPath = "../mirflickr25k/mirflickr/random/"
    ## Extract features from the model
    modelName = "./VGG16WithDistractorModel2"
    model = load_model(modelName)

    newModel = Model(inputs=model.input, outputs=model.get_layer("global_average_pooling2d").output)

    DS_features = extractFeatures(dirPath=DistractorDatasetPath, model=newModel)
    pickle.dump(DS_features, open("DistractorFeatures/DistractorVGG16WithDistractorModel2.txt", 'wb'))