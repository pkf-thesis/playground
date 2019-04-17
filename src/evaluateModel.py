import numpy as np
from evaluator import Evaluator

predictions = np.load("prediction.npy")
truths = np.load("../data/mtat/y_test_pub.npy").astype(int)
evaluator = Evaluator()
evaluator.mean_roc_auc(predictions, truths)
