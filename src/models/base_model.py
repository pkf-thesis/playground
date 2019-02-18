import keras
import numpy as np
from data_generator import DataGenerator


class BaseModel:

    def __init__(self, song_length, dimension, n_channels, num_labels):

        self.song_length = song_length # Length of the song to the network

        self.dimension = dimension
        self.n_channels = n_channels
        self.input_shape = np.empty((*self.dimension, self.n_channels)).shape
        self.num_labels = num_labels

        self.model = self.build_model()
        self.model.summary()

    def transform_data(self, data):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def train(self, train_x, train_y, epoch_size, validation_size=0.1, batch_size=100):

        self.model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

        num_train = len(train_x)

        if validation_size != 0.0:
            validation_x = train_x[:int(num_train*validation_size)]
            validation_y = train_y[:int(num_train*validation_size)]
            train_x = train_x[int(num_train*validation_size):]
            train_y = train_y[int(num_train*validation_size):]

        train_gen = DataGenerator(self.transform_data, train_x, train_y, batch_size,
                                  dim=self.dimension, n_classes=self.num_labels)

        val_gen = DataGenerator(self.transform_data, validation_x, validation_y, batch_size,
                                dim=self.dimension, n_classes=self.num_labels)

        self.model.fit_generator(
            train_gen,
            steps_per_epoch= num_train // batch_size,
            validation_data= val_gen,
            validation_steps=len(validation_x) // batch_size,
            epochs=epoch_size
        )
