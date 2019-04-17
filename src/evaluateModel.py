import numpy as np
from sklearn.metrics import roc_auc_score


def mean_roc_auc(predictions, truths):
    num_predictions = len(predictions)
    n_labels = len(truths[0])
    auc = np.zeros(n_labels)
    label_truths = np.zeros((n_labels, num_predictions))
    label_predictions = np.zeros((n_labels, num_predictions))
    for labelIndex in range(n_labels):
        for index in range(num_predictions):
            label_predictions[labelIndex, index] = predictions[index, labelIndex]
            label_truths[labelIndex, index] = truths[index, labelIndex]
    for i in range(n_labels):
        truths = label_truths[i]
        predictions = label_predictions[i]
        auc[i] = roc_auc_score(truths, predictions)
    return np.mean(auc)


_predictions = np.load("prediction.npy")
_truths = np.load("../data/mtat/y_test_pub.npy").astype(int)
auc_score = mean_roc_auc(_predictions, _truths)
print(auc_score)
