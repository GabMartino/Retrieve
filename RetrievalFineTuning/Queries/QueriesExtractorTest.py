import pickle

if __name__ == "__main__":

    paths = [
        "QueriesDSFeaturesVGG16block2FineTuned1024_0.txt"
    ]
    
    for path in paths:
        features = None
        with open(path, "rb") as f:
            features = pickle.load(f)
            for f in features[:2]:
                print(f[0])

