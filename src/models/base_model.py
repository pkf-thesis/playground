from abc import ABC
from typing import Tuple
import os
from datetime import datetime
import multiprocessing
import numpy as np

import keras
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LambdaCallback

from matplotlib import pyplot as plt

from data_generator import DataGenerator
from utils import utils
from utils.learning_rate_tracker import LearningRateTracker
from utils.roc_auc_callback import ROCAUCCallback


class BaseModel(ABC):

    def __init__(self, song_length: int, dim, n_channels: int, batch_size: int, args):
        self.path = "../sdb/data/%s/%s.npz"

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
            csv_logger = CSVLogger(filename=utils.make_path(args.logging, "%s-%s_%s.csv" % (args.d, self.model_name, datetime.now())))
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

    def train(self, train_x, train_y, valid_x, valid_y, epoch_size, lr, weight_name):
        # Save model
        json_name = 'model_architecture_%s_%s_%s.6f.json' % (self.model_name, self.dataset, lr)
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
            metrics=['accuracy'])

        train_gen = utils.train_generator(train_x, train_y, self.batch_size, 25, self.dimension[0], self.n_labels,
                                          self.dataset, self.path)

        val_gen = DataGenerator(self.transform_data, valid_x, valid_y, batch_size=self.batch_size, n_channels=1,
                                dim=self.dimension, n_classes=self.n_labels)

        check_pointer = ModelCheckpoint(weight_name, monitor='val_loss', verbose=0,
                                        save_best_only=True, mode='auto', save_weights_only=True)
        self.callbacks.append(check_pointer)
        self.callbacks.append(ROCAUCCallback(valid_x, valid_y, self.dimension[0], self.n_labels, self.dataset, self.path))

        history = train_model.fit_generator(
            train_gen,
            callbacks=self.callbacks,
            steps_per_epoch=len(train_x) // self.batch_size * utils.calculate_num_segments(self.dimension[0]),
            # steps_per_epoch=10, # Used for testing
            validation_data=val_gen,
            validation_steps=len(valid_x) // self.batch_size,
            # validation_steps=10, # Used for testing
            epochs=epoch_size,
            workers=self.workers,
            use_multiprocessing=False,
        )

        self._plot_training(history, lr)

        return train_model

    def retrain(self, train_x, train_y, valid_x, valid_y, epoch_size, lr, lr_prev, weight_name):

        train_model = self.model
        if self.gpu:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(self.gpu)
                train_model = multi_gpu_model(self.model, gpus=len(self.gpu))
                use_multiprocessing = True
            except:
                pass

        # load weights model
        splitted_weight_name = weight_name.split("_")
        splitted_weight_name[-1] = str(lr_prev)
        train_model.load_weights("_".join(splitted_weight_name) + ".hdf5")

        train_model.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'])

        train_gen = utils.train_generator(train_x, train_y, self.batch_size, 25, self.dimension[0], self.n_labels,
                                          self.dataset, self.path)

        val_gen = DataGenerator(self.transform_data, valid_x, valid_y, batch_size=self.batch_size, n_channels=1,
                                dim=self.dimension, n_classes=self.n_labels)

        check_pointer = ModelCheckpoint(weight_name, monitor='val_loss', verbose=0,
                                        save_best_only=True, mode='auto', save_weights_only=True)
        self.callbacks.append(check_pointer)
        self.callbacks.append(ROCAUCCallback(valid_x, valid_y, self.dimension[0], self.n_labels, self.dataset, self.path))

        history = train_model.fit_generator(
            train_gen,
            callbacks=self.callbacks,
            steps_per_epoch=len(train_x) // self.batch_size * utils.calculate_num_segments(self.dimension[0]),
            validation_data=val_gen,
            validation_steps=len(valid_x) // self.batch_size,
            epochs=epoch_size,
            workers=self.workers,
            use_multiprocessing=False,
        )

        self._plot_training(history, lr)

        return train_model

    def _plot_training(self, history, lr):
        plot_name = '%s_%s_%s.png' % (self.model_name, lr, self.dataset)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('acc_' + plot_name, bbox_inches='tight')
        plt.clf()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss_' + plot_name, bbox_inches='tight')
        plt.clf()
