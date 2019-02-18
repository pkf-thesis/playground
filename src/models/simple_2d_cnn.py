from models.base_model import BaseModel

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization


class Simple2DCNN(BaseModel):

    def build_model(self, input_shape, labels):
        model = Sequential()

        # First conv block
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Second conv block
        model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Third conv block
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Fourth conv block
        model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Fifth conv block
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
        model.add(Dropout(0.25))

        # MLP
        model.add(Flatten())
        model.add(Dense(labels, activation='softmax'))