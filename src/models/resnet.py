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

from keras.layers.merge import add
from keras import backend as K
from keras.regularizers import l2

from utils.utils import calculate_num_segments
from models.base_model import BaseModel


class ResNet(BaseModel):

    model_name = "resnet"

    input_dim = 3 * 3 ** 9
    overlap = 0

    def transform_data(self, ids_temp, labels_temp, batch_size: int):
        num_segments = calculate_num_segments(self.input_dim)
        new_batch_size = batch_size * num_segments

        # Initialization
        x = np.empty((new_batch_size, *self.dimension, self.n_channels), dtype='float32')
        y = np.empty((new_batch_size, len(labels_temp[0])))

        count = 0
        # Generate data
        for i, song_id in enumerate(ids_temp):
            song = np.load("../sdb/data/%s/%s.npz" % (self.dataset, song_id))

            song_temp = None
            try:
                song_temp = song['arr_0']
            except:
                print(song_id)

            # Convert song to sub songs
            sub_signals = self.split_song(song_temp, num_segments)

            for sub_song in sub_signals:
                sub_song = sub_song.reshape((-1, 1))
                x[count,] = sub_song
                y[count] = labels_temp[i]

                count += 1

        return x, y

    def _shortcut(self, input, residual):
        channel = 2
        step = 1
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride = int(round(input_shape[step] / residual_shape[step], 0))
        equal_channels = input_shape[channel] == residual_shape[channel]

        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride > 1 or not equal_channels:
            shortcut = Conv1D(residual_shape[channel],
                              kernel_size=1,
                              strides=stride,
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)

        return add([shortcut, residual])

    def build_model(self):
        activ = 'relu'
        init = 'he_uniform'

        pool_input = Input(shape=self.input_shape)

        conv0 = Conv1D(128, 9, strides=9, padding='valid', kernel_initializer=init, name="conv0")(pool_input)
        bn0 = BatchNormalization(name="bn0")(conv0)
        activ0 = Activation(activ, name="activ0")(bn0)

        # First
        conv1 = Conv1D(128, 3, strides=3, padding='valid', kernel_initializer=init, name="conv1")(activ0)
        bn1 = BatchNormalization(name="bn1")(conv1)
        activ1 = Activation(activ, name="activ1")(bn1)

        conv2 = Conv1D(128, 3, strides=3, padding='valid', kernel_initializer=init, name="conv2")(activ1)
        bn2 = BatchNormalization(name="bn2")(conv2)
        activ2 = Activation(activ, name="activ2")(bn2)

        res1 = self._shortcut(activ0, activ2)

        # Second
        conv3 = Conv1D(256, 3, strides=3, padding='valid', kernel_initializer=init, name="conv3")(res1)
        bn3 = BatchNormalization(name="bn3")(conv3)
        activ3 = Activation(activ, name="activ3")(bn3)

        conv4 = Conv1D(256, 3, strides=3, padding='valid', kernel_initializer=init, name="conv4")(activ3)
        bn4 = BatchNormalization(name="bn4")(conv4)
        activ4 = Activation(activ, name="activ4")(bn4)

        res2 = self._shortcut(res1, activ4)

        # Third
        conv5 = Conv1D(256, 3, strides=3, padding='valid', kernel_initializer=init, name="conv5")(res2)
        bn5 = BatchNormalization(name="bn5")(conv5)
        activ5 = Activation(activ, name="activ5")(bn5)

        conv6 = Conv1D(256, 3, strides=3, padding='valid', kernel_initializer=init, name="conv6")(activ5)
        bn6 = BatchNormalization(name="bn6")(conv6)
        activ6 = Activation(activ, name="activ6")(bn6)

        res3 = self._shortcut(res2, activ6)

        # Fourth
        conv7 = Conv1D(512, 3, strides=3, padding='valid', kernel_initializer=init, name="conv7")(res3)
        bn7 = BatchNormalization(name="bn7")(conv7)
        activ7 = Activation(activ, name="activ7")(bn7)

        conv8 = Conv1D(512, 3, strides=3, padding='valid', kernel_initializer=init, name="conv8")(activ7)
        bn8 = BatchNormalization(name="bn8")(conv8)
        activ8 = Activation(activ, name="activ8")(bn8)

        res4 = self._shortcut(res3, activ8)

        Flattened = Flatten()(res4)

        output = Dense(self.n_labels, activation='sigmoid')(Flattened)
        model = Model(input=pool_input, output=output)

        return model

    def split_song(self, song, num_segments):
        # Empty list to hold data
        temp_song = []

        # Get the input songs array size
        x_shape = song.shape[0]
        chunk = self.input_dim

        # Split song and create sub samples
        splitted_song = [song[i * chunk: i * chunk + chunk] for i in range(0, num_segments)]
        for sub_song in splitted_song:
            if len(sub_song) == chunk:
                temp_song.append(sub_song)

        return np.array(temp_song)
