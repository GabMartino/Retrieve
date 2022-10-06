import pickle

from matplotlib import pyplot as plt


def plot3Data(*lists, title):
    for (l, label) in list(lists)[0]:
        plt.plot(l, label=label)

    plt.ylim(bottom=0, top=1)
    plt.xlim(left=1)
    plt.title(title)
    plt.ylabel("map")
    plt.xlabel("k")
    plt.legend(loc='upper right')
    plt.savefig(title+".png")
    plt.show()


if __name__ == "__main__":


    ##PRINT BRUTEFORCE KNN SEARCH ON VGG16 FEATURE WITHOUT DISTRACTOR

    paths = [
        'BruteforceData_DS_featuresVGG19block4FineTuned512_512.txt',

    ]

    with open('BruteforceData/BruteforceData_DS_featuresVGG19block4FineTuned512_512.txt', "rb") as f1:
        a = pickle.load(f1)
        lists = [
            (a, "Finetuning block 5 "),
        ]
        plot3Data(lists,
                  title="Bruteforce FineTuning VGG19")



