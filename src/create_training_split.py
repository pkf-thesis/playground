from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

base_path = "data/%s"

train_ids = [song.rstrip() for song in open(base_path % "mtat/train_path.txt")]
valid_ids = [song.rstrip() for song in open(base_path % "mtat/valid_path.txt")]
test_ids = [song.rstrip() for song in open(base_path % "mtat/test_path.txt")]

all_ids = train_ids + valid_ids + test_ids

y_train_old = np.load(base_path % "mtat/y_train_pub.npy")
y_valid_old = np.load(base_path % "mtat/y_valid_pub.npy")
y_test_old = np.load(base_path % "mtat/y_test_pub.npy")

y_all = np.concatenate((y_train_old, y_valid_old, y_test_old), axis=0)


mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)

i = 0
for train_index, test_index in mskf.split(all_ids, y_all):
    train = open(base_path % "mtat/cross/" + str(i) + "_train.txt", mode="w")
    test = open(base_path % "mtat/cross/" + str(i) + "_test.txt", mode="w")

    for train_id in train_index:
        train.write(all_ids[train_id] + "\n")
    for test_id in test_index:
        test.write(all_ids[test_id] + "\n")

    Y_train, Y_test = y_all[train_index], y_all[test_index]

    np.save(base_path % "mtat/cross/" + str(i) + "_train.npy", Y_train)
    np.save(base_path % "mtat/cross/" + str(i) + "_test.npy", Y_test)

    train.close()
    test.close()

    i += 1
