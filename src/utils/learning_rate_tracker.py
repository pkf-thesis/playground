from keras.callbacks import Callback
from keras import backend as K


class LearningRateTracker(Callback):

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer

        # lr printer
        lr = K.eval(K.variable((optimizer.lr * (1. / (1. + optimizer.decay * K.cast(optimizer.iterations, dtype='float32'))))))
        print('\nEpoch %d, lr: %.6f' % (epoch+1, lr))
