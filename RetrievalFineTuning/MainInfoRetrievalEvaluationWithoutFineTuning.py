import math
import os
import pickle
import random
import sys

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from API.PPIndex import SearchIndex
from API.PivotSelector import PivotSelector
from ExtractFeatures import extractFeatures

sys.setrecursionlimit(10000)
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

def QueryTest(queryDs, index):
    DS_features = None
    with open(queryDs, 'rb') as f:
        DS_features = pickle.load(f)

    Query = random.choice(DS_features)
    k = 10
    p = index.query(Query[2], k=k, z=k)

    #print(Query[0])
    #print([(v[0], v[1]) for v in p])


def Evaluate(Queries, index, limit , bias):
    if bias == None:
        bias = 0
    z = limit
    APsforQuery = []
    for queryClass, queryImagePath, queryFeatures in tqdm(Queries):
        APs = []
        for k in range(1, z+1):
            p = index.query(queryFeatures, k=k, z=k + bias, method="perturbation")

            resultsClasses = [ ResultObjClass for ResultObjClass, ResultObjPath, ResultObjFeatures in p]

            AP = getAveragePrecision(queryClass, resultsClasses)

            APs.append(AP)
        APsforQuery.append(APs)

    ## per ogni valore di k da 1 a z, abbiamo diversi valori di precisione per ogni singola query

    SUMS = np.sum(np.array(APsforQuery), 0)  ## somma per ogni query ogni specifico valore per ogni k
    MAPs = SUMS / len(Queries)  ## normalizza il valore per tutte le query

    return MAPs


def BuildIndex(IndexName, Dataset, NumberOfPivots, PivotsSelectionMethod, l):
    p = PivotSelector(Number_of_pivots=NumberOfPivots, dataset=Dataset, indexOfTuple=2, method=PivotsSelectionMethod,
                      distance="l2")
    pivots = p.getPivots()

    print(" \tCreate new index on those feature set...")
    index = SearchIndex(pivots=pivots, l=l)  ## l should be less than the number of pivots
    index.insertData(Dataset)
    os.makedirs(os.path.dirname(IndexName), exist_ok=True)
    index.dumpIndex(IndexName)
    return index


def fromParamToIndexPath(Dataset, NumPivots, Method):
    path = "./Indexes/Dataset="+str(Dataset)+"/Method="+str(Method)+"/NumPivots="+str(NumPivots)
    return path
def fromParamToExperimentPath(Dataset, NumPivots, Method, Z=None):
    path = ""

    if Z != None:
        path = "./ExperimentsPerturbation/Dataset="+str(Dataset)+"/Method="+str(Method)+"/Z="+str(Z)+"/NumPivots="+str(NumPivots)
    else:
        path = "./ExperimentsPerturbation/Dataset=" + str(Dataset) + "/Method=" + str(Method) + "/NumPivots=" + str(NumPivots)
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
def Experiment(DataName, Data, Queries):
    NumberOfPivots = [50, 100, 200]
    Zs = [50, 100, 200]
    for z in Zs:
        for Np in NumberOfPivots:
            P = Np
            l = 3
            pivotmethod = "Kmedoids"
            Z = z
            indexName = fromParamToIndexPath(DataName,P, pivotmethod)
            print("Getting index...", indexName )
            index = GetIndex(indexName)
            if index == None :
                index = BuildIndex(IndexName=indexName, Dataset = Data, NumberOfPivots=P, PivotsSelectionMethod=pivotmethod,l=l)


            print("Start Evaluation...")
            resultsMap = Evaluate(Queries=Queries, index=index, limit=100, bias = Z)
            Name = fromParamToExperimentPath(DataName,P, pivotmethod, Z=Z)
            os.makedirs(os.path.dirname(Name), exist_ok=True)
            pickle.dump(resultsMap, open(Name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    FeaturesPath = [
        "CombinedVGG16block2FineTuned1024_0.txt",
    ]

    QueriesAssociated = [
        "QueriesDSFeaturesVGG16block2FineTuned1024_0.txt",
    ]
    for dspath, queries in zip(FeaturesPath, QueriesAssociated):
        print("Extracting Data and Queries...")
        dataset = None
        with open("./Combined/" + dspath, "rb") as f:
            dataset = pickle.load(f)

        Queries = None
        with open("./Queries/" + queries, "rb") as f:
            Queries = pickle.load(f)

        print("Start Experiment...")
        Experiment(dspath, dataset, Queries=Queries)





