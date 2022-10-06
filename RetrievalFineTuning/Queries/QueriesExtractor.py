import pickle

if __name__ == "__main__":
    a = None
    with open("./QueriesFileList.txt", "rb") as f:
        a = pickle.load(f)


    paths = [
        "DSFeaturesVGG16WithDistractorModel2.txt"

    ]

    for path in paths:
        Queries = []
        features = None
        with open("../DSFeatures/"+path, "rb") as f:
            features = pickle.load(f)

        for v in features:
            if v[1] in a:
                Queries.append(v)

        pickle.dump(Queries,open("Queries" + path, "wb"))