import numpy as np
import evaluator

_predictions = np.load("../results/predictions_mtat_8e-05.npy")
_truths = np.load("../data/mtat/y_test_pub.npy").astype(int)
auc_score = evaluator.mean_roc_auc(_predictions, _truths)
print(auc_score)
