import pickle
import random

from matplotlib import pyplot as plt
from textwrap import wrap
def getRandomColor():
    color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color
def plot3Data(*lists, title):
    lastGroup = 1
    for (l, label, g) in list(lists)[0]:
        if g == 0:
            plt.plot(l, label=label)
        elif g != lastGroup:
            color = getRandomColor()
            lastGroup = g
            plt.plot(l, label=label, color=color)


    plt.ylim(bottom=0, top=1)
    plt.xlim(left=1)
    plt.title("\n".join(wrap(title,60)))
    plt.ylabel("map")
    plt.xlabel("k")
    plt.legend(loc='upper right')
    plt.savefig(title + ".jpg")

    plt.show()


if __name__ == "__main__":


    paths = [
        './ExactSearchWithDistractor/VGG19block5ds.txt',
        'PP Index - VGG19Num Pivots100l value=10Pivot Selection Method Kmedoidsmultiplier 2bias = 100perturbation',
        'PP Index - VGG19Num Pivots100l value=10Pivot Selection Method Kmedoidsmultiplier 2bias = 100',
    ]
    paths1 = [
        './ExactSearchWithDistractor/VGG19block5ds.txt',
        'PP Index - VGG19Num Pivots100l value=10Pivot Selection Method Kmedoidsmultiplier 2bias = 50perturbation',
        './Experiments2/ChangingNp/KmedoidsPivots/bias=50/PP Index - VGG19Num Pivots100l value=10Pivot Selection Method Kmedoidmultiplier 2bias = 50',
    ]
    data = []
    for path in paths:

        with open(path, "rb") as f:
            data.append(pickle.load(f))

    labels = [
        "Exact Search",
        "with perturbation",
        "without perturbation",
    ]
    groups = [0]*3
    groups1 = [
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3
    ]
    lists = []
    for d, label, g in zip(data, labels, groups):
        lists.append((d, label, g))

    plot3Data(lists[:6],
              title="PP Index VGG19 - Kmedoids Pivots -  l = 10 - z = 2k + 100 perturbation")

