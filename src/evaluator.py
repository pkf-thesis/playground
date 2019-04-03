from typing import List

from models.base_model import BaseModel
from data_generator import DataGenerator
from sklearn.metrics import roc_auc_score
import numpy as np


class Evaluator:

    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def evaluate(self, base_model: BaseModel, model, x_test: List[str], y_test: List[str]) -> None:

        test_generator = DataGenerator(base_model.transform_data, x_test, y_test, batch_size=self.batch_size,
                                  dim=base_model.dimension, n_channels=base_model.n_channels, n_classes=base_model.n_labels)

        score = model.evaluate_generator(test_generator, len(x_test) / 5)
        print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

        # Testing predict
        # song = np.load('../npys/blues.00015.npy')
        # song = song[0:3 * 3 ** 9]
        # song = song.reshape((-1, 1))
        # print(song.shape)
        # song = song.reshape(1, 3 * 3**9, 1)
        # prediction = model.predict(song)
        # print(prediction)

    # Example
    # predictions   = array([[0.54, 0.98, 0.43], [0.32, 0.18, 0.78], [0.78, 0.76, 0.86]])
    # truths        = array([[1, 1, 0], [0, 0, 1], [1, 1, 0]])
    # mean_roc_auc  = 0.66
    def mean_roc_auc(predictions, truths):
        n_predictions = len(predictions)
        auc = np.zeros(n_predictions)
        for index in range(n_predictions):
            prediction = predictions[index]
            truth = truths[index]
            auc[index] = roc_auc_score(truth, prediction)
        return np.mean(auc)