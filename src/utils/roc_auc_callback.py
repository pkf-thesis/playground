from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import numpy as np

from utils import utils


class ROCAUCCallback(Callback):

    def __init__(self, x_val, y_val, sample_length, num_labels, dataset, path):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.sample_length = sample_length
        self.num_labels = num_labels
        self.path = path
        self.dataset = dataset

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        num_segments = utils.calculate_num_segments(self.sample_length)

        x_val_temp = np.zeros((num_segments, self.sample_length, 1))
        x_val = np.zeros((len(self.x_val), self.num_labels))

        for i, song_id in enumerate(self.x_val):
            song = np.load(self.path % (self.dataset, song_id))['arr_0']

            for segment in range(0, num_segments):
                x_val_temp[segment] = song[segment * self.sample_length:
                                           segment * self.sample_length + self.sample_length].reshape((-1, 1))

            x_val[i] = np.mean(self.model.predict(x_val_temp), axis=0)

        auc = np.zeros(len(self.x_val))

        for index in range(len(self.x_val)):
            prediction = x_val[index]
            truth = self.y_val[index]
            auc[index] = roc_auc_score(truth, prediction)

        print('\rroc-auc_val: %s' % (str(np.mean(auc))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
