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


class SampleCNNDeepResNet(BaseModel):

    model_name = "SampleCNN_deep_resnet"

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
        stride = int(round(input_shape[step] / residual_shape[step]))
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

    def generate_stacked_convolutions(self, input_layer, filters):
        activation = 'relu'
        init = 'he_uniform'
        output_layer = input_layer
        for i in range(3):
            conv = Conv1D(filters, 3, padding='same', kernel_initializer=init)(output_layer)
            bn = BatchNormalization()(conv)
            activ = Activation(activation)(bn)
            output_layer = activ
        return self._shortcut(input_layer, output_layer)

    def build_model(self):
        activ = 'relu'
        init = 'he_uniform'

        pool_input = Input(shape=self.input_shape)

        conv0 = Conv1D(128, 3, strides=3, padding='valid', kernel_initializer=init, name="conv0")(pool_input)
        bn0 = BatchNormalization(name="bn0")(conv0)
        activ0 = Activation(activ, name="activ0")(bn0)

        deep1 = self.generate_stacked_convolutions(activ0, 128)
        MP1 = MaxPooling1D(pool_size=3)(deep1)

        deep2 = self.generate_stacked_convolutions(MP1, 128)
        MP2 = MaxPooling1D(pool_size=3)(deep2)

        deep3 = self.generate_stacked_convolutions(MP2, 256)
        MP3 = MaxPooling1D(pool_size=3)(deep3)

        deep4 = self.generate_stacked_convolutions(MP3, 256)
        MP4 = MaxPooling1D(pool_size=3)(deep4)

        deep5 = self.generate_stacked_convolutions(MP4, 256)
        MP5 = MaxPooling1D(pool_size=3)(deep5)

        deep6 = self.generate_stacked_convolutions(MP5, 256)
        MP6 = MaxPooling1D(pool_size=3)(deep6)

        deep7 = self.generate_stacked_convolutions(MP6, 256)
        MP7 = MaxPooling1D(pool_size=3)(deep7)

        deep8 = self.generate_stacked_convolutions(MP7, 256)
        MP8 = MaxPooling1D(pool_size=3)(deep8)

        deep9 = self.generate_stacked_convolutions(MP8, 512)
        MP9 = MaxPooling1D(pool_size=3)(deep9)

        deep10 = self.generate_stacked_convolutions(MP9, 512)
        dropout1 = Dropout(0.5)(deep10)

        Flattened = Flatten()(dropout1)

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
        splitted_song = [song[i*chunk: i*chunk+chunk] for i in range(0, num_segments)]
        for sub_song in splitted_song:
            if len(sub_song) == chunk:
                temp_song.append(sub_song)

        return np.array(temp_song)
