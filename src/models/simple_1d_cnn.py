from models.base_model import BaseModel
import utils.gtzan_genres as gtzan

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten


class Simple1DCNN(BaseModel):

    model_name = "Simple1D_CNN"

    def transform_data(self, ids_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dimension, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, id in enumerate(ids_temp):
            # Store sample
            x = np.load('../npys/' + id)
            x = x.reshape((-1, 1))
            X[i, ] = x

            # Store class
            y[i] = gtzan.genres[self.labels[i]]

        return X, y

    def build_model(self):
        model = Sequential()

        # First conv block
        model.add(Conv1D(16, kernel_size=3, strides=1, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(0.25))

        # Second conv block
        model.add(Conv1D(32, 3, strides=1, activation='relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(0.25))

        # Third conv block
        model.add(Conv1D(64, 3, strides=1, activation='relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(0.25))

        # MLP
        model.add(Flatten())
        model.add(Dense(labels, activation='softmax'))

        return model