import itertools
import pickle
from functools import partial
from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

from API.PivotSelector import PivotSelector
from API.PrefixTree import IntegerTrie


class SearchIndex:

    def __init__(self, pivots, l, distance="l2"):
        self.pivots = pivots
        if l > len(pivots):
            str(KeyError("l value should be < |P|"))
            exit()

        if distance not in ["l2", "cosine"]:
            str(KeyError("Distance type not present."))
            exit()
        self.l = l
        self.distance = distance
        self.trie = IntegerTrie()


    def _getDistance(self, a, b):
        if self.distance == "l2":
            return np.linalg.norm(a - b)
        else:
            return dot(a, b) / (norm(a) * norm(b))

    def extractPermutation3(self, featuresObject, query=False):
        from bisect import bisect_right
        objPermutations = []
        distances = []

        def insertPivot(pivotTuple):
            distance = pivotTuple[0]
            i = bisect_right(a=distances, x=distance)
            objPermutations.insert(i, pivotTuple[1])
            distances.insert(i, distance)

        pool = Pool()
        results = pool.map(partial(self._getDistance, featuresObject),self.pivots)
        print(results)
        exit()
        for idx, d in  enumerate(pool.imap(partial(self._getDistance, featuresObject),self.pivots)):
            t = (d, idx)
            insertPivot(t)

        pool.close()
        if not query:
            objPermutations = objPermutations[:self.l]
        else:
            objPermutations = objPermutations[:len(self.trie.root.children)]
        return objPermutations



    def extractPermutation2(self, featuresObject, query=False):
        from bisect import bisect_right
        objPermutations = []
        distances = []
        def insertPivot(pivotTuple):
            distance = pivotTuple[0]
            i = bisect_right(a=distances, x=distance)
            objPermutations.insert(i, pivotTuple[1])
            distances.insert(i, distance)

        for idx, p in enumerate(self.pivots):
            t = (self._getDistance(featuresObject, p), idx)
            insertPivot(t)

        if not query:
            objPermutations = objPermutations[:self.l]
        else:
            objPermutations = objPermutations[:len(self.trie.root.children)]
        return objPermutations


    def extractPermutation(self, featuresObject, query=False):
        distances_to_pivots = []
        for p in self.pivots:
            distances_to_pivots.append(self._getDistance(featuresObject, p))


        objPermutations = sorted(range(len(distances_to_pivots)), key=lambda k: distances_to_pivots[k])

        if not query:
            objPermutations = objPermutations[:self.l]
        else:
            objPermutations = objPermutations[:len(self.trie.root.children)]
        return objPermutations


    def insertData(self, data):
        '''
            From a set of pivots and a set of data:
            1. calculate the distance between data points and pivots
            2. order pivots distances for each data point o1 = (p4,p5,p0,...)

        '''
        print("Creating permutations...")
        objectData = []
        for ObjClass, ObjPath, ObjFeatures in tqdm(data):
            ##List of distances respect to pivots
            ObjPermutations = self.extractPermutation(ObjFeatures)

            ## Data are made by (path, feature, permutation)
            objectData.append((ObjClass, ObjPath, ObjFeatures, ObjPermutations))


        self.trie.insertData(objectData)



    def dumpIndex(self, path):
        pickle.dump(self, open(path, "wb"))
    '''
        Use the prefix tree to capture z objects, than reorder according to the real distances and fetch the first k objects
    
    '''
    def query(self, q, k, z, method = "standard" ):

        ## Fetch z object
        if z == k:
            z += 1
        perm = self.extractPermutation2(q, query=True)
        objects = self.trie.knnquery(perm, k=z, method=method)
        assert len(objects) >= z, ["%d" % len(objects),"%d" %z]
        objects = sorted(objects, key=lambda x: self._getDistance(x[2], q))
        if self._getDistance(q, objects[0][2]) == 0:
            objects = objects[1:]

        return objects[:k]


if __name__ == "__main__":
   pass