from typing import List, Tuple
import numpy as np
from keras import Input, Model

from models.base_model import BaseModel

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization

from keras.layers.merge import add
from keras import backend as K

from utils.utils import calculate_num_segments


class SampleCNN39(BaseModel):

    model_name = "SampleCNN_3_9"

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

    def build_model_2(self):
        activ = 'relu'
        init = 'he_uniform'

        pool_input = Input(shape=(self.input_dim, 1))

        conv0 = Convolution1D(128, 3, subsample_length=3, border_mode='valid', init=init, name="conv0")(pool_input)
        bn0 = BatchNormalization(name="bn0")(conv0)
        activ0 = Activation(activ, name="activ0")(bn0)

        conv1 = Convolution1D(128, 3, border_mode='same', init=init, name="conv1")(activ0)
        bn1 = BatchNormalization()(conv1)
        activ1 = Activation(activ)(bn1)
        MP1 = MaxPooling1D(pool_length=3)(activ1)

        conv2 = Convolution1D(128, 3, border_mode='same', init=init)(MP1)
        bn2 = BatchNormalization()(conv2)
        activ2 = Activation(activ)(bn2)
        MP2 = MaxPooling1D(pool_length=3)(activ2)

        conv3 = Convolution1D(256, 3, border_mode='same', init=init)(MP2)
        bn3 = BatchNormalization()(conv3)
        activ3 = Activation(activ)(bn3)
        MP3 = MaxPooling1D(pool_length=3)(activ3)

        conv4 = Convolution1D(256, 3, border_mode='same', init=init)(MP3)
        bn4 = BatchNormalization()(conv4)
        activ4 = Activation(activ)(bn4)
        MP4 = MaxPooling1D(pool_length=3)(activ4)

        conv5 = Convolution1D(256, 3, border_mode='same', init=init)(MP4)
        bn5 = BatchNormalization()(conv5)
        activ5 = Activation(activ)(bn5)
        MP5 = MaxPooling1D(pool_length=3)(activ5)

        conv6 = Convolution1D(256, 3, border_mode='same', init=init)(MP5)
        bn6 = BatchNormalization()(conv6)
        activ6 = Activation(activ)(bn6)
        MP6 = MaxPooling1D(pool_length=3)(activ6)

        conv7 = Convolution1D(256, 3, border_mode='same', init=init)(MP6)
        bn7 = BatchNormalization()(conv7)
        activ7 = Activation(activ)(bn7)
        MP7 = MaxPooling1D(pool_length=3)(activ7)

        conv8 = Convolution1D(256, 3, border_mode='same', init=init)(MP7)
        bn8 = BatchNormalization()(conv8)
        activ8 = Activation(activ)(bn8)
        MP8 = MaxPooling1D(pool_length=3)(activ8)

        conv9 = Convolution1D(512, 3, border_mode='same', init=init)(MP8)
        bn9 = BatchNormalization()(conv9)
        activ9 = Activation(activ)(bn9)
        MP9 = MaxPooling1D(pool_length=3)(activ9)

        conv10 = Convolution1D(512, 1, border_mode='same', init=init)(MP9)
        bn10 = BatchNormalization()(conv10)
        activ10 = Activation(activ)(bn10)
        dropout1 = Dropout(0.5)(activ10)

        Flattened = Flatten()(dropout1)

        output = Dense(self.n_labels, activation='sigmoid')(Flattened)
        model = Model(input=pool_input, output=output)

        return model

    def build_model(self):
        model = Sequential()

        #First layer
        model.add(Conv1D(128, kernel_size=3, strides=3, padding='valid', activation='relu', input_shape=self.input_shape))
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
        model.add(Dense(self.n_labels, activation='sigmoid'))

        return model