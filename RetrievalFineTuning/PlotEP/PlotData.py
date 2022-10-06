import pickle
from textwrap import wrap

from matplotlib import pyplot as plt


if __name__ == "__main__":

    DataName = "EPMeans_numPivots50"
    Data = None
    with open(DataName, "rb") as f:
        Data = pickle.load(f)

    print(Data)
    Z = [50, 100, 200]
    plt.plot(Z, Data, marker= "o", label="NumPivots = 50")

    DataName = "EPMeans_numPivots100"
    Data = None
    with open(DataName, "rb") as f:
        Data = pickle.load(f)

    plt.plot(Z, Data, marker="o", label="NumPivots = 100")

    DataName = "EPMeans_numPivots200"
    Data = None
    with open(DataName, "rb") as f:
        Data = pickle.load(f)

    plt.plot(Z, Data, marker="o", label="NumPivots = 200")
    plt.xlabel("bias, Z = K + bias")
    plt.ylabel("EP")
    plt.title(" Error on Position on Z - with Perturbation ")
    plt.legend()
    plt.show()