import itertools
import pickle
import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import dot
from numpy.linalg import norm
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm


class PivotSelector:

    def __init__(self, Number_of_pivots, dataset, indexOfTuple, method="3n", distance="l2"):
        self.n = Number_of_pivots
        self.ds = dataset
        if method not in ["random", "3n", "Kmedoids"]:
            raise ValueError("The inserted method does not exist.")
        self.method = method
        self.indexOfTuple = indexOfTuple
        if distance not in ["l2", "cosine"]:
            raise ValueError("The inserted distance type does not exist.")
        self.distance = distance

    def _getDistance(self, a, b):
        if self.distance == "l2":
            return np.linalg.norm(a - b)
        else:
            return dot(a, b) / (norm(a) * norm(b))

    def _randomObjects(self):
        set = random.choices(self.ds, k=self.n)
        pivots = [x[self.indexOfTuple] for x in set]
        return pivots

    def _furthestFromAnObject(self, r, set):
        maxDist = 0
        furthestIdxObj = None
        for idx, o in enumerate(set):
            d = self._getDistance(o, r)
            if d > maxDist:
                maxDist = d
                furthestIdxObj = idx
        return set[furthestIdxObj]

    def _3nAlgorithm(self):

        pivots = []
        sample = random.choices(self.ds, k=3 * self.n)
        sample = [x[self.indexOfTuple] for x in sample] ## get a 3n sample
        idx = random.choice(range(len(sample)))
        r = sample[idx]
        pivots.append(self._furthestFromAnObject(r, sample))## this is the first pivot
        t =  sample
        t.pop(idx)
        pivots.append(self._furthestFromAnObject(pivots[0], t))##second pivot removing the first random object

        ## Save all distance from the object to the pivots
        distances = []
        for o in sample:
            distancesToPivots = []
            for p in pivots:
                distancesToPivots.append(self._getDistance(o,p))
            distances.append(distancesToPivots)
        #print(distances)
        pbar = tqdm(initial=2, total=self.n)
        while len(pivots) < self.n:
            ## Find the object more distant to the other pivot
            max = 0
            index = None
            for idx, d in enumerate(distances):
                minDistance = min(d)
                if minDistance > max:
                    max = minDistance
                    index = idx
            for idx, o in enumerate(sample):
                distanceToNewPivot = self._getDistance(o, sample[index])
                distances[idx].append(distanceToNewPivot)

            ## insert the object to the set of pivots
            pivots.append(sample[index])
            pbar.update(1)
        pbar.close()
        return pivots

    def Kmedoid(self):
        vectors = [ x[self.indexOfTuple] for x in self.ds]
        kmedoids = KMedoids(n_clusters=self.n, random_state=0, init="random").fit(vectors)
        return kmedoids.cluster_centers_


    def getPivots(self):
        if self.method == "random":
            return self._randomObjects()
        elif self.method == "3n":
            return self._3nAlgorithm()
        elif self.method == "Kmedoids":
            return self.Kmedoid()



if __name__ == "__main__":
    DS_features = None
    with open('../RetrievalNoFineTuning/DSFeatures/old/DS_featuresVGG16Normalized.txt', 'rb') as f:
        DS_features = pickle.load(f)

    p = PivotSelector(Number_of_pivots = 500, dataset=DS_features, indexOfTuple=2, method="efficiency")

    ##Test pivots
    pivots = p.getPivots()
    exit()
    distances = []
    for (a, b) in list(itertools.product(pivots, pivots)):
        d = np.linalg.norm(a - b)
        distances.append(d)
    plt.hist(distances, bins=100, range=[0, 2])
    plt.ylabel("distances")
    plt.show()