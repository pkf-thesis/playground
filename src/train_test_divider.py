import os

def splitData(path, trainRatio):
    ids = os.listdir(path)
    trainIndex = int(len(ids) * trainRatio)
    train = ids[:trainIndex]
    test = ids[trainIndex:]

    return (train, test)