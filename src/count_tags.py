import numpy as np

train = np.load("../data/mtat/y_train_pub.npy")
valid = np.load("../data/mtat/y_valid_pub.npy")
test = np.load("../data/mtat/y_test_pub.npy")

for i in range(0, 50):
    print(np.sum(train[..., i]))

print("\n")

for i in range(0, 50):
    print(np.sum(valid[..., i]))

print("\n")

for i in range(0, 50):
    print(np.sum(test[:, i]))


