import argparse

import numpy as np
from keras.utils import multi_gpu_model

from models.sample_cnn_3_9 import SampleCNN39
import evaluator as evaluator

batch_size = 25

parser = argparse.ArgumentParser()
parser.add_argument("-d", "-data", help="gtzan, mtat or msd")
parser.add_argument("-logging", help="Logs to csv file")
parser.add_argument("-gpu", type=list, help="Run on gpu's, and which")
parser.add_argument("-local", help="Whether to run local or on server")

args = parser.parse_args()

base_path = "../data/mtat/"
x_test = [song.rstrip() for song in open(base_path + "test_path.txt")]
y_test = np.load(base_path + "y_test_pub.npy")

base_model = SampleCNN39(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size,
                         weight_name='../results/best_weights_%s_%s.hdf5', args=args)

model = multi_gpu_model(base_model.model, gpus=2)

print("Testing")
x_pred = evaluator.predict(base_model, model, x_test, None)

'Save predictions'
np.save("../results/predictions_baseline.npy", x_pred)

test_result = evaluator.mean_roc_auc(x_pred, y_test)
print("Mean ROC-AUC: %s" % test_result)