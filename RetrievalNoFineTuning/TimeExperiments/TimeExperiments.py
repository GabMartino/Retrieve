



'''

    Check increasing of the recall with Z
    at a fixed value of pivots and K
    comparing also the time


'''
import os
import pickle

import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

def _getDistance( a, b, distance="l2"):
    if distance == "l2":
        return np.linalg.norm(a - b)
    else:
        return dot(a, b) / (norm(a) * norm(b))

def Recall(TruePositives, Retrieved):

  
    if len(TruePositives) != len(Retrieved):
        exit()

    counter = 0
    for v in Retrieved:
        if v in TruePositives:
            counter += 1

    recall = counter/len(TruePositives)

    return recall


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
def Experiment(dataset, Queries, K, Z, index):

    Stats = []
    import time
    for queryClass, queryImagePath, queryFeatures in tqdm(Queries):
        start_time = time.time()
        exactResults, _ = NaiveKNNResearch(queryFeatures,dataset, K, [] )
        exactExecutionTime = time.time() - start_time
        start_time = time.time()
        p = index.query(queryFeatures, k=K, z=K + Z, method="perturbation")
        approximateExecutionTime = time.time() - start_time
        exactresultsObjPaths = [ ResultObjPath for ResultObjClass, ResultObjPath, ResultObjFeatures in exactResults]
        approxresultsObjPaths = [ResultObjPath for ResultObjClass, ResultObjPath, ResultObjFeatures in p]
        Stats.append((exactExecutionTime, approximateExecutionTime, Recall(exactresultsObjPaths, approxresultsObjPaths)))

    return Stats
def fromParamToIndexPath(Dataset, NumPivots, Method, l=None):
    path = ""
    if l == None:
        path = "../Indexes/Dataset="+str(Dataset)+"/Method="+str(Method)+"/NumPivots="+str(NumPivots)
    else:
        path = "../Indexes/Dataset="+str(Dataset)+"/Method="+str(Method)+"/l="+str(l)+"/NumPivots="+str(NumPivots)
    return path
def GetIndex(IndexNamePath):
    try:
        index = None
        os.makedirs(os.path.dirname(IndexNamePath), exist_ok=True)
        with open(IndexNamePath, 'rb') as i:
            index = pickle.load(i)
        return index
    except:
        return None
if __name__ ==  "__main__":
    FeaturesPath = [
        "CombinedVGG16block3_pool.txt",
        "CombinedVGG16block4_pool.txt",
        "CombinedVGG16block5_pool.txt",
        "CombinedVGG19block3_pool.txt",
        "CombinedVGG19block4_pool.txt",
        "CombinedVGG19block5_pool.txt",
    ]

    QueriesAssociated = [
        "QueriesDS_featuresVGG16block3_pool.txt",
        "QueriesDS_featuresVGG16block4_pool.txt",
        "QueriesDS_featuresVGG16block5_pool.txt",
        "QueriesDS_featuresVGG19block3_pool.txt",
        "QueriesDS_featuresVGG19block4_pool.txt",
        "QueriesDS_featuresVGG19block5_pool.txt",
    ]
    for dspath, queries in zip(FeaturesPath[1:2], QueriesAssociated[1:2]):
        print("Extracting Data and Queries...")
        dataset = None
        with open("../CombinedFeatures/" + dspath, "rb") as f:
            dataset = pickle.load(f)

        Queries = None
        with open("../Queries/" + queries, "rb") as f:
            Queries = pickle.load(f)

        NumPivots = 50
        indexName = fromParamToIndexPath(dspath, NumPivots, Method="Kmedoids", l=3)
        index = GetIndex(indexName)
        print("Start Experiment...")
        Zs = [0, 20, 50, 100, 200, 400, 500]
        K = 20
        for z in Zs:
            Z = z
            Stats = Experiment(dataset, Queries=Queries, K=K, Z=Z, index=index )

            pickle.dump(Stats, open("StatsTimesperturbation_l=3K="+str(K)+"Z="+str(Z)+"NumPivots= "+str(NumPivots)+".txt", "wb"))
