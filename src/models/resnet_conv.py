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


class ResNetConv(BaseModel):

    model_name = "ResNetConv"

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
            shortcut = Conv1D(filters=residual_shape[channel],
                            kernel_size=1,
                            strides=stride,
                            padding="valid",
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2(0.0001))(input)

        return add([shortcut, residual])

    def build_model(self):
        activ = 'relu'
        init = 'he_uniform'

        pool_input = Input(shape=(self.input_shape))

        conv0 = Convolution1D(128, 3, subsample_length=3, padding='valid', kernel_initializer=init, name="conv0")(pool_input)
        bn0 = BatchNormalization(name="bn0")(conv0)
        activ0 = Activation(activ, name="activ0")(bn0)

        conv1 = Convolution1D(128, 3, padding='valid', kernel_initializer=init, name="conv1")(activ0)
        bn1 = BatchNormalization()(conv1)
        activ1 = Activation(activ)(bn1)

        conv2 = Convolution1D(128, 3, padding='valid', kernel_initializer=init)(activ1)
        bn2 = BatchNormalization()(conv2)
        activ2 = Activation(activ)(bn2)

        residual1 = self._shortcut(activ0, activ2)

        conv3 = Convolution1D(256, 3, padding='valid', kernel_initializer=init)(residual1)
        bn3 = BatchNormalization()(conv3)
        activ3 = Activation(activ)(bn3)

        conv4 = Convolution1D(256, 3, padding='valid', kernel_initializer=init)(activ3)
        bn4 = BatchNormalization()(conv4)
        activ4 = Activation(activ)(bn4)

        residual2 = self._shortcut(residual1, activ4)

        conv5 = Convolution1D(256, 3, padding='valid', kernel_initializer=init)(residual2)
        bn5 = BatchNormalization()(conv5)
        activ5 = Activation(activ)(bn5)

        conv6 = Convolution1D(256, 3, padding='valid', kernel_initializer=init)(activ5)
        bn6 = BatchNormalization()(conv6)
        activ6 = Activation(activ)(bn6)

        residual3 = self._shortcut(residual2, activ6)

        conv7 = Convolution1D(256, 3, padding='valid', kernel_initializer=init)(residual3)
        bn7 = BatchNormalization()(conv7)
        activ7 = Activation(activ)(bn7)

        conv8 = Convolution1D(512, 3, padding='valid', kernel_initializer=init)(activ7)
        bn8 = BatchNormalization()(conv8)
        activ8 = Activation(activ)(bn8)

        residual4 = self._shortcut(residual3, activ8)

        conv9 = Convolution1D(512, 3, padding='valid', kernel_initializer=init)(residual4)
        bn9 = BatchNormalization()(conv9)
        activ9 = Activation(activ)(bn9)

        conv10 = Convolution1D(512, 1, padding='same', kernel_initializer=init)(activ9)
        bn10 = BatchNormalization()(conv10)
        activ10 = Activation(activ)(bn10)
        dropout1 = Dropout(0.5)(activ10)

        Flattened = Flatten()(dropout1)

        output = Dense(self.n_labels, activation='sigmoid')(Flattened)
        model = Model(input=pool_input, output=output)

        return model
