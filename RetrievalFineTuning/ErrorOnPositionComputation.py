import os
import pickle

import numpy as np
from numpy import dot
from numpy.linalg import norm

from RetrievalFineTuning.MainInfoRetrievalEvaluationWithoutFineTuning import fromParamToIndexPath, GetIndex


def _getDistance( a, b, distance="l2"):
    if distance == "l2":
        return np.linalg.norm(a - b)
    else:
        return dot(a, b) / (norm(a) * norm(b))





def Evaluate(Queries, Data, index, k, bias):

    EPmean = 0
    for q in Queries:
        p = index.query(q[2], k=k, z=k + bias, method="perturbation")
        resultsIndex = NaiveKNNResearch(q[2], Data )
        EP = 0
        for j, o in enumerate(p):
            for i, idx in enumerate(resultsIndex):
                if Data[idx][1] == o[1]:
                    EP += i - j

        EP /= len(Data)*len(p)
        EPmean += EP
    EPmean /= len(Queries)
    print(EPmean)
    return EPmean


'''
    Return the positions of the objects ordered respect to the query
'''
def NaiveKNNResearch(query, dataset):

    distances = []
    #Calcolo tutte le distanze
    for o in dataset:
        #print(len(o[2]), len(query))
        d = _getDistance(query, o[2], distance="l2")
        distances.append(d)

    #ordino le distanze per valore di distanza rispetto alla query, ma estraggo gli indici
    orderedIndexes = sorted(range(len(distances)), key=lambda k: distances[k])

    return  orderedIndexes

def Experiment(DataName, Data, Queries):
    print("Experiment:")
    NumberOfPivots = [50, 100, 200]
    Zs = [50, 100, 200]

    for Np in NumberOfPivots:
        EPMeans = []
        for z in Zs:
            P = Np
            l = 3
            pivotmethod = "Kmedoids"
            Z = z
            indexName = fromParamToIndexPath(DataName,P, pivotmethod)
            print("Getting index...", indexName )
            index = GetIndex(indexName)
            print("Start Evaluation...")
            k = 50
            EPMean = Evaluate(Queries, Data=Data, index=index, k = k, bias=z)
            EPMeans.append(EPMean)

        pickle.dump(EPMeans,open("EPMeans_numPivots"+str(Np), 'wb'), protocol=pickle.HIGHEST_PROTOCOL )



if __name__ == "__main__":

    print("Fetch Dataset and Queries...")
    FeaturesPath = [
        "CombinedVGG16block2FineTuned1024_0.txt",
    ]

    QueriesAssociated = [
        "QueriesDSFeaturesVGG16block2FineTuned1024_0.txt",
    ]
    for dspath, queries in zip(FeaturesPath, QueriesAssociated):
        dataset = None
        with open("./Combined/" + dspath, "rb") as f:
            dataset = pickle.load(f)

        Queries = None
        with open("./Queries/" + queries, "rb") as f:
            Queries = pickle.load(f)

        print(len(dataset[0][2]), len(Queries[0][2]))
        print("Start Experiment...")
        Experiment(dspath, dataset, Queries=Queries)