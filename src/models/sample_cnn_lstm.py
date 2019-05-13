from typing import List, Tuple
import numpy as np
from keras import Input, Model

from models.base_model import BaseModel

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization

from utils.utils import calculate_num_segments


class SampleCNNLSTM(BaseModel):

    model_name = "SampleCNN_LSTM"

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
        model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))

        model.add(Dense(self.n_labels, activation='sigmoid'))

        return model
