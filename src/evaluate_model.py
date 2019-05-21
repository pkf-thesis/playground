import numpy as np
import evaluator

_predictions = np.load("../../mixed_pooling/predictions_SampleCNN_deep_resnet_mtat_1.6e-05.npy")
_truths = np.load("../data/mtat/y_test_pub.npy").astype(int)
auc_score = evaluator.individual_roc_auc(_predictions, _truths)

for tag in auc_score:
    print(tag)
