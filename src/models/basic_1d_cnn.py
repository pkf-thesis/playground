from typing import List, Tuple

from models.base_model import BaseModel
import utils.gtzan_genres as gtzan

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten


class Basic1DCNN(BaseModel):

    model_name = "Basic1D_CNN"

    def transform_data(self, ids_temp: List[str], batch_size: int) -> Tuple[np.array, np.array]:
        # Initialization
        X = np.empty((batch_size, *self.dimension, self.n_channels))
        y = np.empty(batch_size, dtype=int)

        # Generate data
        for i, id in enumerate(ids_temp):
            genre = id.split('.')[0]

            # Store sample
            x = np.load('../npys/' + id)
            x = x.reshape((-1, 1))
            X[i, ] = x

            # Store class
            y[i] = gtzan.genres[genre]

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
        model.add(Dense(self.n_labels, activation='softmax'))

        return model