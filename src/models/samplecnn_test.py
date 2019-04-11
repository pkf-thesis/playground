from typing import List, Tuple
import numpy as np
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization

def build_model():
    model = Sequential()

    #First layer
    model.add(Conv1D(128, kernel_size=3, strides=3, padding='valid', activation='relu', input_shape=(59049,1)))
    model.add(BatchNormalization())

    #Second layer
    model.add(Conv1D(128, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Third layer
    model.add(Conv1D(128, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Fourth layer
    model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Fith layer
    model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Sixth layer
    model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Seventh layer
    model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Eigth layer
    model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Ninth layer
    model.add(Conv1D(512, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Tenth layer
    model.add(Conv1D(512, 3, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=3))

    #Eleventh layer
    model.add(Conv1D(512, 1, strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    #Twelveth layer
    model.add(Flatten())
    model.add(Dense(50, activation='sigmoid'))

    return model