from abc import ABC
from typing import Tuple
import os
import multiprocessing

import keras
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
from data_generator import DataGenerator

from utils import utils


class BaseModel(ABC):

    def __init__(self, song_length: int, dim, n_channels: int, n_labels: int, args):
        self.callbacks = []
        self.gpu = None
        self.song_length = song_length # Length of the song to the network

        self.dimension = dim
        self.n_channels = n_channels
        self.input_shape = np.empty((*self.dimension, self.n_channels)).shape
        self.n_labels = n_labels

        self.model = self.build_model()
        self.model.summary()

        weight_name = 'best_weights_%s_%s.6f.hdf5' % (self.model_name, self.dimension)
        check_pointer = ModelCheckpoint(weight_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto',
                                        save_weights_only=True)

        if args.logging:
            csv_logger = CSVLogger(filename=utils.make_path(args.logging,self.model_name + '.csv'))
            self.callbacks.append(csv_logger)

        if args.gpu:
            self.gpu = args.gpu

        self.callbacks.append(check_pointer)

    @property
    def model_name(self):
        raise NotImplementedError

    def transform_data(self, data: str, batch_size: int) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train(self, train_x, train_y, epoch_size, validation_size=0.1, batch_size=100) -> None:
        use_multiprocessing = False
        if self.gpu:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(self.gpu)
                self.model = multi_gpu_model(self.model, gpus=len(self.gpu))
                use_multiprocessing = True
            except:
                pass

        self.model.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['categorical_accuracy'])

        num_train = len(train_x)

        if validation_size != 0.0:
            validation_x = train_x[:int(num_train*validation_size)]
            validation_y = train_y[:int(num_train*validation_size)]
            train_x = train_x[int(num_train*validation_size):]
            train_y = train_y[int(num_train*validation_size):]

        train_gen = DataGenerator(self.transform_data, train_x, train_y, batch_size,
                                  dim=self.dimension, n_classes=self.n_labels)

        val_gen = DataGenerator(self.transform_data, validation_x, validation_y, batch_size,
                                dim=self.dimension, n_classes=self.n_labels)

        num_train = len(train_x)

        workers = multiprocessing.cpu_count()
        print('Using ' + str(workers) + ' workers')

        self.model.fit_generator(
            train_gen,
            callbacks=self.callbacks,
            steps_per_epoch=num_train // batch_size,
            validation_data=val_gen,
            validation_steps=len(validation_x) // batch_size,
            epochs=epoch_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
