import pickle
from textwrap import wrap

from matplotlib import pyplot as plt


def plot3Data(*lists, title):
    for (l, label) in list(lists)[0]:
        plt.plot(l, label=label)


    plt.ylim(bottom=0, top=1)
    plt.xlim(left=1)
    plt.title("\n".join(wrap(title,60)))
    plt.ylabel("map")
    plt.xlabel("k")
    plt.legend(loc='upper right')
    plt.savefig(title + ".jpg")

    plt.show()


if __name__ == "__main__":


    VGG16Paths = [
        "exactSearchmAPCombinedVGG16block3_pool.txt",
        "exactSearchmAPCombinedVGG16block4_pool.txt",
        "exactSearchmAPCombinedVGG16block5_pool.txt"
    ]
    VGG19Paths = [
        "exactSearchmAPCombinedVGG19block3_pool.txt",
        "exactSearchmAPCombinedVGG19block4_pool.txt",
        "exactSearchmAPCombinedVGG19block5_pool.txt"
    ]

    datasVGG16 = []

    for path in VGG16Paths:
        data = None
        with open(path, "rb") as f:
            data = pickle.load(f)
        datasVGG16.append(data)

    datasVGG19 = []

    for path in VGG19Paths:
        data = None
        with open(path, "rb") as f:
            data = pickle.load(f)
        datasVGG19.append(data)

    listsVGG16 = [
        (datasVGG16[0], "Exact Search from block 3"),
        (datasVGG16[1], "Exact Search from block 4"),
        (datasVGG16[2], "Exact Search from block 5")
    ]
    listsVGG19 = [
        (datasVGG19[0], "Exact Search from block 3"),
        (datasVGG19[1], "Exact Search from block 4"),
        (datasVGG19[2], "Exact Search from block 5")
    ]
    plot3Data(listsVGG16,
              title="BruteForce VGG16 250 Queries ")
    plot3Data(listsVGG19,
              title="BruteForce VGG19 250 Queries ")
