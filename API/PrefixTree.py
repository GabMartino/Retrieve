import math
import pickle
import statistics

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class TrieNode:


    def __init__(self, v= None, level=0, data=None, parent=None):
        self.value = v
        self.data = []
        self.numberOfDataObjects = 0
        self.children = []
        self.level = level
        self.parent = parent

    def insert(self, value, data):##value is the permutation
        ## if the value at that index is not present between the children


        if len(value) >= 1 and value[0] not in [c.value for c in self.children] :## the first element of the new permutation to insert is already in the children
            ## create a new child node with that value
            self.numberOfDataObjects += 1
            n = TrieNode(v=value[0], level=self.level+1, parent=self)
            n.insert(value[1:], data)
            #if len(value) == 1:
            #    n.numberOfDataObjects += 1
            #    n.setData(data)
            self.children.append(n)
        else:
            if len(self.children) == 0:
                self.setData(data)
                return
            for child in self.children:##search in children
                if child.value == value[0]:
                    self.numberOfDataObjects += 1 ##update the actual node with a new object, the insertion cannot fail
                    child.insert(value[1:], data)
                    break

    '''
        From the paper:
        "searching for the longest prefix match in the prefix tree whose subtree points to at least z candidate objects."
        we use k in this case
    
    '''

    def searchNode(self, q , k, data):
        #print("ingresso", self.value, q, data)
        if len(self.children) == 0:
            #print("prelevo")
            self.dfs(data, limit=k)
            return

        qtemp = q.copy()
        if len(qtemp) == 0:
            #print(self.value, "prelevo")
            self.dfs(data, limit=k)
            return
        mainCheck = False
        while len(data) < k and len(qtemp) > 0:
            check = False
            for child in self.children:
                if (len(qtemp) >= 1 and child.value == qtemp[0]) or (
                        type(
                            qtemp) == int and child.value == qtemp):
                    check = True
                    mainCheck = True
                    child.searchNode(qtemp[1:], k, data)
                    #print("esco", self.value, qtemp, data, check)
                    break

            #print("esco1", self.value, qtemp, data, check)
            if not check and not mainCheck and  self.value != None:

                #vs = [child.value for child in self.children]
                #print("if not check", self.value, "prelevo", vs, qtemp , check)
                self.dfs(data, limit=k)
                return

            qtemp = qtemp[1:]

    def knnquery(self, q, k, data):
        #if self.numberOfDataObjects > k:##go deep in the tree to find the best match of the subtree
        check = False
        for child in self.children:
            if ((len(q) >= 1 and child.value == q[0]) or (type(q) == int and child.value == q)) and child.numberOfDataObjects > k :##found a node with specific value of the permutation index

                child.knnquery(q[1:], k, data)##go deeper
                check = True

        if not check: ## not found in the children other best match point

            self.dfs(data, limit=k)##retrieve al necessary data possible in the all subtree


        #elif self.numberOfDataObjects <= k:
        #    self.dfs(data, limit=k)


    def setData(self, data):
        self.data.append(data)

    def dfs(self, data, limit=None):
        pointer = 0
        lenght = len(self.children)
        while len(data) < limit:
            if lenght > 0 and pointer < lenght:
                self.children[pointer].dfs(data, limit)
                pointer += 1
            else:
                data.extend(self.data[:limit - len(data)])
                return
        '''
        
        if limit is not None and len(data) < limit and len(self.children) == 0 and self.data not in data: ## if I have a limit and it is still unreached and i have not children
            data.extend(self.data)## object node add value to the data
        elif limit is None and len(self.children) == 0 and self.data not in data:
            data.extend(self.data)
        else:
            for child in self.children:
                child.dfs(data, limit)
        
        '''
    def printNode(self):

        '''

        for _ in range(self.level):
            print("\t", end = '')

        print(" level:" + str(self.level))

        '''
        for _ in range(self.level):
            print("\t", end = '')
        print(str(self.value))


        for _ in range(self.level):
            print("\t", end = '')
        
        
        print(" number of dataobject:" + str(self.numberOfDataObjects))

        for c in self.children:
            c.printNode()

    def print(self):
        '''
        for _ in range(self.level):
            print("\t", end = '')

        print(" level:" + str(self.level))
        '''
        for _ in range(self.level):
            print("\t", end = '')
        print(str(self.value))

        if len(self.data) > 0:
            for _ in range(self.level):
                print("\t", end = '')
            print(" data:"+ str(self.data))


        for c in self.children:
            c.print()


class IntegerTrie:


    def __init__(self):

        self.root = TrieNode(level=0)## only the root has empty value

    def insert(self, value, data ):

        self.root.insert(value, data)

    def insertData(self, data):
        print("Inserting data into Trie")
        for ObjClass, ObjPath, ObjFeatures, ObjPermutations in tqdm(data):
            ##Use permutations as value to create prefix, path and feature are the data
            self.insert(ObjPermutations, (ObjClass, ObjPath, ObjFeatures))

    def dfs(self, limit=None):
        data = []
        self.root.dfs(data, limit=limit)

        return data## to flatten


    def printTrie(self):
        self.root.print()

    def printTrieStructure(self):
        self.root.printNode()

    '''
        Query lenght could be smaller of bigger than the depth of the tree
    '''
    def knnquery(self, q, k, method="standard"):
        data = []
        if method == "standard":
            self.root.knnquery(q, k, data)
        elif method == "perturbation":
            self.root.searchNode(q, k, data)

        return data



    def _statsOnLevel(self, nodeList):
        if len(nodeList) <= 0:
            return (0, 0)

        numOfObjects = []
        for c in nodeList:
            numOfObjects.append(c.numberOfDataObjects)
        meanValue = sum(numOfObjects) / len(numOfObjects)
        std = statistics.stdev(numOfObjects)
        ci = (1.96)*std / math.sqrt(len(numOfObjects))

        return (meanValue, ci)
    def extractStats(self):

        stats = []
        values = self._statsOnLevel(self.root.children)
        print("Statistics on level 1", values)
        stats.append(values)
        childrenTemp = self.root.children
        count = 2
        while len(childrenTemp) > 0:
            subchildren = []
            for c in childrenTemp:
                subchildren.append(c.children)
            subchildren = [item for sublist in subchildren for item in sublist]
            values = self._statsOnLevel(subchildren)
            print("Statistics on level "+str(count), values)
            childrenTemp = subchildren
            count += 1
            stats.append(values)

        return stats

if __name__ == "__main__":

    
    t = IntegerTrie()
    t.insert([4, 5, 6, 8], "ciao1")
    t.insert([5, 1, 2, 6], "ciao2")
    t.insert([3, 2, 8, 4], "ciao3")

    t.insert([7, 1, 2, 6], "ciao4")
    t.insert([7, 1, 2, 6], "ciao4sfgsdfg")
    t.insert([7, 1, 2, 6], "ciao4dfgdfg")
    t.insert([7, 1, 2, 6], "ciaodfgdfg4")


    t.insert([8, 3, 2, 1], "ciao5")
    t.insert([3, 2, 8, 6], "ciao7")
    t.insert([7, 2, 5, 8], "ciao6")

    '''
   
                              |  4   |    5   |    3                |   7               |               8
                        [5] |           [1]         [2]        |        [1] |  [2]          |            [3]
                    [6] |               [2]         [8]         |    [2]  |         [5]        |        [2]
                [8] |                   [6]     |    [4][6]     |     [6]  |             [8]      |       [1]
                ciao1               ciao2          [ciao3, ciao7]  ciao4           ciao6               ciao5
    
    
    '''

    t.printTrieStructure()
    print(t.knnquery(q=[5, 1, 2], k=3))


