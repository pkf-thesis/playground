from typing import List, Tuple
import numpy as np
from keras import Input, Model

from models.base_model import BaseModel

from keras.layers import Dense, Activation, Convolution1D, AveragePooling1D
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import concatenate

from utils.utils import calculate_num_segments

from utils.MixedMaxAvgPooling1D import MixedMaxAvgPooling1D


class MaxAverageNet(BaseModel):

    model_name = "max_average_net_3"

    input_dim = 3 * 3 ** 9
    overlap = 0

    def transform_data(self, ids_temp: List[str], labels_temp, batch_size: int) -> Tuple[np.array, np.array]:
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

    def max_average_pooling(self, input_layer, pool_size=3):
        max_pooling = MaxPooling1D(pool_size)(input_layer)
        average_pooling = AveragePooling1D(pool_size)(input_layer)
        return concatenate([max_pooling, average_pooling])

    def build_model(self):
        activ = 'relu'
        init = 'he_uniform'

        pool_input = Input(shape=(self.input_dim, 1))

        conv0 = Conv1D(128, kernel_size=3, strides=3, padding='valid', kernel_initializer=init, name="conv0")(pool_input)
        bn0 = BatchNormalization(name="bn0")(conv0)
        activ0 = Activation(activ, name="activ0")(bn0)

        conv1 = Conv1D(128, 3, padding='same', kernel_initializer=init, name="conv1")(activ0)
        bn1 = BatchNormalization()(conv1)
        activ1 = Activation(activ)(bn1)
        input_dim1 = self.input_dim / 3
        max_average1 = MixedMaxAvgPooling1D(name='mixmaxavg1', alpha=None, method='layer', input_dim=input_dim1, pool_size=3)(activ1)

        conv2 = Conv1D(128, 3, padding='same', kernel_initializer=init)(max_average1)
        bn2 = BatchNormalization()(conv2)
        activ2 = Activation(activ)(bn2)
        input_dim2 = input_dim1 / 3
        max_average2 = MixedMaxAvgPooling1D(name='mixmaxavg2', alpha=None, method='layer', input_dim=input_dim2, pool_size=3)(activ2)

        conv3 = Conv1D(256, 3, padding='same', kernel_initializer=init)(max_average2)
        bn3 = BatchNormalization()(conv3)
        activ3 = Activation(activ)(bn3)
        input_dim3 = input_dim2 / 3
        max_average3 = MixedMaxAvgPooling1D(name='mixmaxavg3', alpha=None, method='layer', input_dim=input_dim3, pool_size=3)(activ3)

        conv4 = Conv1D(256, 3, padding='same', kernel_initializer=init)(max_average3)
        bn4 = BatchNormalization()(conv4)
        activ4 = Activation(activ)(bn4)
        input_dim4 = input_dim3 / 3
        max_average4 = MixedMaxAvgPooling1D(name='mixmaxavg4', alpha=None, method='layer', input_dim=input_dim4, pool_size=3)(activ4)

        conv5 = Conv1D(256, 3, padding='same', kernel_initializer=init)(max_average4)
        bn5 = BatchNormalization()(conv5)
        activ5 = Activation(activ)(bn5)
        input_dim5 = input_dim4 / 3
        max_average5 = MixedMaxAvgPooling1D(name='mixmaxavg5', alpha=None, method='layer', input_dim=input_dim5, pool_size=3)(activ5)

        conv6 = Conv1D(256, 3, padding='same', kernel_initializer=init)(max_average5)
        bn6 = BatchNormalization()(conv6)
        activ6 = Activation(activ)(bn6)
        input_dim6 = input_dim5 / 3
        max_average6 = MixedMaxAvgPooling1D(name='mixmaxavg6', alpha=None, method='layer', input_dim=input_dim6, pool_size=3)(activ6)

        conv7 = Conv1D(256, 3, padding='same', kernel_initializer=init)(max_average6)
        bn7 = BatchNormalization()(conv7)
        activ7 = Activation(activ)(bn7)
        input_dim7 = input_dim6 / 3
        max_average7 = MixedMaxAvgPooling1D(name='mixmaxavg7', alpha=None, method='layer', input_dim=input_dim7, pool_size=3)(activ7)

        conv8 = Conv1D(512, 3, padding='same', kernel_initializer=init)(max_average7)
        bn8 = BatchNormalization()(conv8)
        activ8 = Activation(activ)(bn8)
        input_dim8 = input_dim7 / 3
        max_average8 = MixedMaxAvgPooling1D(name='mixmaxavg8', alpha=None, method='layer', input_dim=input_dim8, pool_size=3)(activ8)

        conv9 = Conv1D(512, 3, padding='same', kernel_initializer=init)(max_average8)
        bn9 = BatchNormalization()(conv9)
        activ9 = Activation(activ)(bn9)
        input_dim9 = input_dim8 / 3
        max_average9 = MixedMaxAvgPooling1D(name='mixmaxavg9', alpha=None, method='layer', input_dim=input_dim9, pool_size=3)(activ9)

        conv10 = Conv1D(512, 1, padding='same', kernel_initializer=init)(max_average9)
        bn10 = BatchNormalization()(conv10)
        activ10 = Activation(activ)(bn10)
        dropout1 = Dropout(0.5)(activ10)

        Flattened = Flatten()(dropout1)

        output = Dense(self.n_labels, activation='sigmoid')(Flattened)
        model = Model(inputs=pool_input, outputs=output)

        return model
