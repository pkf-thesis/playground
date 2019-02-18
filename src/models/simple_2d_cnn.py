from models.base_model import BaseModel
import utils.gtzan_genres as gtzan

import numpy as np
import librosa

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten


class Simple2DCNN(BaseModel):

    def transform_data(self, ids_temp, batch_size, dim, n_channels):

        batch_size = 19*batch_size

        # Initialization
        X = np.empty((batch_size, *dim, n_channels))
        y = np.empty(batch_size, dtype=int)

        count = 0
        # Generate data
        for i, id in enumerate(ids_temp):
            x = np.load('../npys/' + id)
            genre = id.split('.')[0]

            # Convert song to sub songs
            sub_signals = self.split_song(x)

            for song in sub_signals:

                # map transformation of input songs to melspectrogram using log-scale
                mel_spectogram = librosa.feature.melspectrogram(song, n_fft=1024, hop_length=512)[:, :, np.newaxis]
                mel_spectogram = np.array(list(mel_spectogram))

                X[count, ] = mel_spectogram
                y[count] = gtzan.genres[genre]

                count += 1

        return X, y

    def build_model(self):
        model = Sequential()

        # First conv block
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=self.input_shape))
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
        model.add(Dense(self.num_labels, activation='softmax'))

        return model

    def split_song(self, song, window=0.1, overlap=0.5):
        # Empty list to hold data
        temp_song = []

        # Get the input songs array size
        x_shape = song.shape[0]
        chunk = int(x_shape * window)
        offset = int(chunk * (1. - overlap))

        # Split song and create sub samples
        splitted_song = [song[i:i + chunk] for i in range(0, x_shape - chunk + offset, offset)]
        for sub_song in splitted_song:
            if len(sub_song) == chunk:
                temp_song.append(sub_song)

        return np.array(temp_song)