from __future__ import absolute_import
from __future__ import print_function

import keras
from keras import backend as K


class LossLearningRateScheduler(keras.callbacks.History):
    """
    A learning rate scheduler that relies on changes in loss function
    value to dictate whether learning rate is decayed or not.
    LossLearningRateScheduler has the following properties:
    base_lr: the starting learning rate
    lookback_epochs: the number of epochs in the past to compare with the loss function at the current epoch to determine if progress is being made.
    decay_threshold / decay_multiple: if loss function has not improved by a factor of decay_threshold * lookback_epochs, then decay_multiple will be applied to the learning rate.
    spike_epochs: list of the epoch numbers where you want to spike the learning rate.
    spike_multiple: the multiple applied to the current learning rate for a spike.
    """

    def __init__(self, base_lr, lookback_epochs, spike_epochs=None, spike_multiple=10, decay_threshold=0.002,
                 decay_multiple=0.5, loss_type='val_loss'):

        super(LossLearningRateScheduler, self).__init__()

        self.base_lr = base_lr
        self.lookback_epochs = lookback_epochs
        self.spike_epochs = spike_epochs
        self.spike_multiple = spike_multiple
        self.decay_threshold = decay_threshold
        self.decay_multiple = decay_multiple
        self.loss_type = loss_type

    def on_epoch_begin(self, epoch, logs=None):

        if len(self.epoch) > self.lookback_epochs:

            current_lr = K.get_value(self.model.optimizer.lr)

            target_loss = self.history[self.loss_type]

            last_loss = target_loss[-1]

            change_lr = False
            for i in range(2, self.lookback_epochs):
                if last_loss > target_loss[-i]:
                    change_lr = True
                    break
                last_loss = target_loss[-i]

            if change_lr:

                print(' '.join(
                    ('Changing learning rate from', str(current_lr), 'to', str(current_lr * self.decay_multiple))))
                K.set_value(self.model.optimizer.lr, current_lr * self.decay_multiple)
                current_lr = current_lr * self.decay_multiple

            else:

                print(' '.join(('Learning rate:', str(current_lr))))

            if self.spike_epochs is not None and len(self.epoch) in self.spike_epochs:
                print(' '.join(
                    ('Spiking learning rate from', str(current_lr), 'to', str(current_lr * self.spike_multiple))))
                K.set_value(self.model.optimizer.lr, current_lr * self.spike_multiple)

        else:

            print(' '.join(('Setting learning rate to', str(self.base_lr))))
            K.set_value(self.model.optimizer.lr, self.base_lr)

        return K.get_value(self.model.optimizer.lr)


def main():
    return


if __name__ == '__main__':
    main()

