from typing import List, Tuple
import numpy as np

from models.base_model import BaseModel
import utils.gtzan_genres as gtzan


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten

class SampleCNN39(BaseModel):

    model_name = "SampleCNN_3_9"

    input_dim = 3 * 3 ** 9
    overlap = 512

    def transform_data(self, ids_temp: List[str], batch_size: int) -> Tuple[np.array, np.array]:
        new_batch_size = batch_size * (self.song_length / self.overlap)
        # Initialization
        X = np.empty((new_batch_size, *self.dimension, self.n_channels))
        y = np.empty(new_batch_size, dtype=int)

        count = 0
        # Generate data
        for i, id in enumerate(ids_temp):
            song = np.load('../npys/' + id)
            genre = id.split('.')[0]

            # Convert song to sub songs
            sub_signals = self.split_song(song)

            for sub_song in sub_signals:
                X[count,] = sub_song
                y[count] = gtzan.genres[genre]

                count += 1

        return X, y

    def build_model(self):
        model = Sequential()

        #First layer
        model.add(Conv1D(128, kernel_size=3, strides=3, padding='same', activation='relu', input_shape=self.input_shape))

        #Second layer
        model.add(Conv1D(128, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Third layer
        model.add(Conv1D(128, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Fourth layer
        model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Fith layer
        model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Sixth layer
        model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Seventh layer
        model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Eigth layer
        model.add(Conv1D(256, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Ninth layer
        model.add(Conv1D(512, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Tenth layer
        model.add(Conv1D(512, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=3, strides=3))

        #Eleventh layer
        model.add(Conv1D(512, 1, strides=1, padding='same', activation='relu'))
        model.add(Dropout(0.5))

        #Twelveth layer
        model.add(Flatten())
        model.add(Dense(self.n_labels, activation='sigmoid'))

        return model

    def split_song(self, song):
        # Empty list to hold data
        temp_song = []

        # Get the input songs array size
        x_shape = song.shape[0]
        chunk = self.input_dim
        offset = 512

        # Split song and create sub samples
        splitted_song = [song[i:i + chunk] for i in range(0, x_shape, offset)]
        for sub_song in splitted_song:
            if len(sub_song) == chunk:
                temp_song.append(sub_song)

        return np.array(temp_song)
