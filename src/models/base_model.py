from abc import ABC
from typing import Tuple
import os
import multiprocessing
import numpy as np

import keras
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from matplotlib import pyplot as plt

from data_generator import DataGenerator
from utils import utils
from utils.loss_learning_rate_scheduler import LossLearningRateScheduler
from utils.learning_rate_tracker import LearningRateTracker


class BaseModel(ABC):

    def __init__(self, song_length: int, dim, n_channels: int, batch_size: int, args):
        self.song_length = song_length
        self.dimension = dim
        self.n_channels = n_channels
        self.input_shape = np.empty((*self.dimension, self.n_channels)).shape
        self.n_labels = 10 if args.d == 'gtzan' else 50
        self.batch_size = batch_size

        self.model = self.build_model()
        self.model.summary()

        self.workers = multiprocessing.cpu_count()
        print('Using ' + str(self.workers) + ' workers')

        # Callbacks
        self.callbacks = []
        self.callbacks.append(LearningRateTracker())
        self.callbacks.append(EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto'))
        if args.logging:
            csv_logger = CSVLogger(filename=utils.make_path(args.logging,self.model_name + '.csv'))
            self.callbacks.append(csv_logger)

        self.gpu = None
        if args.gpu:
            self.gpu = args.gpu

        self.dataset = args.d

    @property
    def model_name(self):
        raise NotImplementedError

    def transform_data(self, data: str, batch_size: int) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train(self, train_x, train_y, valid_x, valid_y, epoch_size, lr) -> None:

        # Save model
        json_name = 'model_architecture_%s_%s.6f.json' % (self.model_name, lr)
        if os.path.isfile(json_name) != 1:
            json_string = self.model.to_json()
            open(json_name, 'w').write(json_string)

        use_multiprocessing = False
        train_model = self.model
        if self.gpu:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(self.gpu)
                train_model = multi_gpu_model(self.model, gpus=len(self.gpu))
                use_multiprocessing = True
            except:
                pass

        train_model.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['categorical_accuracy'])

        train_gen = DataGenerator(self.transform_data, train_x, train_y, batch_size=self.batch_size, n_channels=1,
                                  dim=self.dimension, n_classes=self.n_labels)

        val_gen = DataGenerator(self.transform_data, valid_x, valid_y, batch_size=self.batch_size, n_channels=1,
                                dim=self.dimension, n_classes=self.n_labels)

        weight_name = 'best_weights_%s_%s_%s.hdf5' % (self.model_name, self.dimension, lr)
        check_pointer = ModelCheckpoint(weight_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto',
                                        save_weights_only=True)
        self.callbacks.append(check_pointer)

        history = train_model.fit_generator(
            train_gen,
            callbacks=self.callbacks,
            steps_per_epoch=len(train_x) // self.batch_size,
            validation_data=val_gen,
            validation_steps=len(valid_x) // self.batch_size,
            epochs=epoch_size,
            workers=self.workers,
            use_multiprocessing=use_multiprocessing,
        )

        self._plot_training(history, lr)

        return train_model

    def _plot_training(self, history, lr):
        plot_name = '%s_%s_%s.png' % (self.model_name, lr, self.dataset)
        # summarize history for accuracy
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc_' + plot_name, bbox_inches='tight')

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss_' + plot_name, bbox_inches='tight')

        plt.clf()
