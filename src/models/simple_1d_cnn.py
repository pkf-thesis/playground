from src.models.base_model import BaseModel

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization


class Simple1DCNN(BaseModel):

    def build_model(self, input_shape, labels):
        model = Sequential()

        # First conv block
        model.add(Conv1D(16, kernel_size=3, strides=1, activation='relu', input_shape=input_shape))
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