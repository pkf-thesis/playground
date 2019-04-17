from keras.callbacks import Callback
import numpy as np

from utils import utils
import evaluator as evaluator


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
        x_pred = np.zeros((len(self.x_val), self.num_labels))

        for i, song_id in enumerate(self.x_val):
            song = np.load(self.path % (self.dataset, song_id))['arr_0']

            for segment in range(0, num_segments):
                x_val_temp[segment] = song[segment * self.sample_length:
                                           segment * self.sample_length + self.sample_length].reshape((-1, 1))

            x_pred[i] = np.mean(self.model.predict(x_val_temp), axis=0)

        auc = evaluator.mean_roc_auc(x_pred, self.y_val)

        print('\r roc-auc_val: %s' % (str(np.mean(auc))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
