import os
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


def NaiveKNNResearch(query, dataset, k, orderedIndexes):

    if len(orderedIndexes) == 0:
        distances = []
        #Calcolo tutte le distanze
        for o in dataset:
            #print(len(o[2]), len(query))
            d = _getDistance(query, o[2], distance="l2")
            distances.append(d)

        #ordino le distanze per valore di distanza rispetto alla query, ma estraggo gli indici
        orderedIndexes = sorted(range(len(distances)), key=lambda k: distances[k])

    results = []

    for i in orderedIndexes[:k+1]:##remove the first always
        results.append(dataset[i])
    if (results[0][2] == query).all():
        results = results[1:]
    return results[:k], orderedIndexes


def Evaluate( dataset, Queries):

    z = 101
    APsforQuery = []
    for queryClass, queryImagePath, queryFeatures in tqdm(Queries):
        APs = []
        orderedIndexes = []
        for k in range(1, z):
            p,orderedIndexes = NaiveKNNResearch(queryFeatures, dataset, k=k, orderedIndexes=orderedIndexes)
            resultsClasses = [ResultObjClass for ResultObjClass, ResultObjPath, ResultObjFeatures in p]
            AP = getAveragePrecision(queryClass, resultsClasses)
            APs.append(AP)
        APsforQuery.append(APs)

    ## per ogni valore di k da 1 a z, abbiamo diversi valori di precisione per ogni singola query

    SUMS = np.sum(np.array(APsforQuery), 0)  ## somma per ogni query ogni specifico valore per ogni k
    MAPs = SUMS / len(Queries)  ## normalizza il valore per tutte le query


    return MAPs



if __name__ == "__main__":


    FeaturesPath = [
        "CombinedVGG16block3_pool.txt",
        "CombinedVGG16block4_pool.txt",
        "CombinedVGG16block5_pool.txt",
        "CombinedVGG19block3_pool.txt",
        "CombinedVGG19block4_pool.txt",
        "CombinedVGG19block5_pool.txt",
    ]

    QueriesAssociated =[
        "QueriesDS_featuresVGG16block3_pool.txt",
        "QueriesDS_featuresVGG16block4_pool.txt",
        "QueriesDS_featuresVGG16block5_pool.txt",
        "QueriesDS_featuresVGG19block3_pool.txt",
        "QueriesDS_featuresVGG19block4_pool.txt",
        "QueriesDS_featuresVGG19block5_pool.txt",
    ]
    for dspath, queries in zip(FeaturesPath[3:], QueriesAssociated[3:]):
        dataset = None
        with open("../CombinedFeatures/"+dspath, "rb") as f:
            dataset = pickle.load(f)

        Queries = None
        with open("../Queries/"+queries, "rb") as f:
            Queries = pickle.load(f)

        print(len(dataset[0][2]), len(Queries[0][2]))
        MAPs = Evaluate(dataset=dataset, Queries=Queries)
        pickle.dump(MAPs, open("exactSearchmAP"+dspath, 'wb'))

