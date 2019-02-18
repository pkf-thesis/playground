import os

def splitData(path, trainRatio):
    ids = os.listdir(path)
    trainIndex = int(len(ids) * trainRatio)
    train_x = ids[:trainIndex]
    test_x = ids[trainIndex:]
    train_y = list(map(lambda id: id.split(".")[0], train_x))
    test_y = list(map(lambda id: id.split(".")[0], test_x))

    return (train_x, train_y, test_x, test_y)