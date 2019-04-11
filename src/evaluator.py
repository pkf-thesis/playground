from typing import List

from sklearn.metrics import roc_auc_score
import numpy as np

from models.base_model import BaseModel
from data_generator import DataGenerator
from utils import utils


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

    def predict(self, base_model: BaseModel, model, x_test: List[str], lr):
        'Load best weights'
        model.load_weights(base_model.weight_name % (base_model.model_name, lr))

        sample_length = base_model.dimension[0]
        num_segments = utils.calculate_num_segments(sample_length)

        x_test_temp = np.zeros((num_segments, sample_length, 1))
        x_pred = np.zeros((len(x_test), base_model.n_labels))

        for i, song_id in enumerate(x_test):
            song = np.load(base_model.path % (base_model.dataset, song_id))['arr_0']

            for segment in range(0, num_segments):
                x_test_temp[segment] = song[segment * sample_length:
                                            segment * sample_length + sample_length].reshape((-1, 1))

            x_pred[i] = np.mean(model.predict(x_test_temp), axis=0)

        return x_pred

    # Example
    # predictions   = array([[0.54, 0.98, 0.43], [0.32, 0.18, 0.78], [0.78, 0.76, 0.86]])
    # truths        = array([[1, 1, 0], [0, 0, 1], [1, 1, 0]])
    # mean_roc_auc  = 0.66
    def mean_roc_auc(self, predictions, truths):
        num_predictions = len(predictions)
        auc = np.zeros(num_predictions)

        for index in range(num_predictions):
            prediction = predictions[index]
            truth = truths[index]
            auc[index] = roc_auc_score(truth, prediction)

        return np.mean(auc)
