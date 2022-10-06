import pickle
import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
from tqdm import tqdm

from RetrievalNoFineTuning.ExtractFeatures import extractFeatures


def _getDistance( a, b, distance="l2"):
    if distance == "l2":
        return np.linalg.norm(a - b)
    else:
        return dot(a, b) / (norm(a) * norm(b))

def NaiveKNNResearch(query, dataset, k):
    distances = []
    #Calcolo tutte le distanze
    for o in dataset:
        d = _getDistance(query, o[2], distance="l2")
        distances.append(d)

    #ordino le distanze per valore di distanza rispetto alla query, ma estraggo gli indici
    indexes = sorted(range(len(distances)), key=lambda k: distances[k])
    results = []

    for i in indexes[1:k+1]:##remove the first always
        results.append(dataset[i])
    return results
'''
    Precision = | relevant document | ^ | retrieved Document | / |Retrieved document|
        


'''

def getAveragePrecision(Truevalue, results):


    rank = [ True if r == Truevalue else False for r in results ]

    numOfTrue = 0
    precision = []
    for idx, v in enumerate(rank):
        if v:
            numOfTrue += 1
            precision.append(numOfTrue / (idx + 1))

    if numOfTrue == 0:
        return 0

    return sum(precision)/numOfTrue


def getRecall():
    pass

def Evaluate( featureSetPath,Num_of_queries, k  ):
    #Num_of_queries = 50
    DS_features = None
    with open(featureSetPath, 'rb') as f:
        DS_features = pickle.load(f)

    Queries = random.choices(population=DS_features, k=Num_of_queries)

    z = k
    APsforQuery = []
    for queryClass, queryImagePath, queryFeatures in tqdm(Queries):
        APs = []
        for k in range(1, z):
            p = NaiveKNNResearch(queryFeatures, DS_features, k=k)
            resultsClasses = [ ResultObjClass for ResultObjClass, ResultObjPath, ResultObjFeatures in p]
            AP = getAveragePrecision(queryClass, resultsClasses)
            APs.append(AP)
        APsforQuery.append(APs)

    ## per ogni valore di k da 1 a z, abbiamo diversi valori di precisione per ogni singola query

    SUMS = np.sum(np.array(APsforQuery), 0)  ## somma per ogni query ogni specifico valore per ogni k
    MAPs = SUMS / len(Queries)  ## normalizza il valore per tutte le query


    pickle.dump(MAPs, open("BruteforceData_" + featureSetPath, 'wb'))


if __name__ == "__main__":



    #paths = ["DS_featuresVGG19Normalized.txt",
     #        "DS_featuresVGG19block4_poolNormalized.txt",
     #        "DS_featuresVGG19block3_poolNormalized.txt"]
    paths = ["DS_featuresVGG19block4FineTuned512_512.txt"]

    #for path in paths:
    Evaluate(paths[0],50, k = 100)


