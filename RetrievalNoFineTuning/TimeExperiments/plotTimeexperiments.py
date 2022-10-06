import math
import pickle
import random

import numpy
import rpy2
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr

#utils = rpackages.importr('utils')
#utils.install_packages("reticulate")
#utils.chooseCRANmirror(ind=1)
# Install packages
#packnames = ('mratios')
#utils.install_packages(packnames)

# Load packages
mratios = importr('mratios')
import numpy as np
from matplotlib import pyplot as plt

def plotStatsOfLevel(values, level, color, title):
    mean = values[0]
    ci = values[1]

    left = level - 1
    right = level + 1
    top = None
    bottom = None
    if isinstance(ci, numpy.ndarray):
        top = mean + ci[1]
        bottom = mean - ci[0]
    else:
        top = mean + ci
        bottom = mean - ci

    plt.plot([level, level], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)

    plt.plot(level, mean, 'o', color=color)
    plt.title(title)


def getFillerRatioCI(data1, data2):
    from rpy2 import robjects

    htest = mratios.ttestratio(robjects.FloatVector(data2),robjects.FloatVector(data1))
    df = robjects.conversion.rpy2py(htest)

    CI = numpy.asarray(df[3])
    mean = numpy.asarray(df[4])[2]
    print(numpy.asarray(df[4]))
    return mean, CI

def getMeanValueAndCI(data):

    mean = np.mean(data)
    std = np.std(data)

    N = len(data)
    return mean, 1.96*std/math.sqrt(N)
def getRandomColor():
    color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color
if __name__ =="__main__":
    K = 20
    NumPivots = 50
    Zs = [0, 20, 50, 100, 200, 400, 500]
    for z in Zs:
        Z = z
        Name = "StatsTimesperturbation_l=3K="+str(K)+"Z="+str(Z)+"NumPivots= "+str(NumPivots)+".txt"

        data = None
        with open(Name, "rb") as f:
            data = pickle.load(f)
        data1 = [v[0] for v in data]
        data2 = [v[1] for v in data]
        mean, ci = getFillerRatioCI(data1, data2)

        plotStatsOfLevel((mean*100, ci*100), Z, getRandomColor(), title="Time ratio between NaiveSearch and PPIndex")
        plt.xlabel("B , Z = K + b")
        plt.ylabel("Percentage of the Naive Search")
        plt.ylim(top=15)
    plt.savefig("Time ratio between NaiveSearch and PPIndex with perturbation.jpg")
    plt.show()

    for z in Zs:
        Z = z
        Name = "StatsTimesperturbation_l=3K=" + str(K) + "Z=" + str(Z) + "NumPivots= " + str(NumPivots) + ".txt"
        if Z == None:
            Z = 0
        data = None
        with open(Name, "rb") as f:
            data = pickle.load(f)
        #mean, ci = getMeanValueAndCIOnTwoSets(data1, data2)
        recallValues = [v[2] for v in data]
        #plotStatsOfLevel((mean, ci), Z, getRandomColor(), title="Time ratio between NaiveSearch and PPIndex")
        print(recallValues)
        mean, ci = getMeanValueAndCI(recallValues)

        plotStatsOfLevel((mean, ci), Z, getRandomColor(), title="Recall on Z")
        plt.xlabel("B , Z = K + b")
        plt.ylabel("Percentage of the Naive Search")
    plt.savefig("Recall on Z with perturbation.jpg")
    plt.show()