import numpy as np
import keras
from sklearn.utils import shuffle


class DataGenerator(keras.utils.Sequence):

    def __init__(self, transform, ids, labels, batch_size, dim, n_channels, n_classes, num_segments=1, song_batch=1, shuffle_data=True):
        self.transform_data = transform
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_segments = num_segments
        self.song_batch = song_batch
        self.shuffle_data = shuffle_data

        self.x_indices = None
        self.y_indices = None
        self.song_batch_index = 0

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.ids) / (self.batch_size*self.num_segments)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        print(index)
        # Load batch of songs
        if index-1 % self.song_batch == 0:
            # Generate indexes of the song batch
            x_indexes = range(self.song_batch_index * self.song_batch, (self.song_batch_index + 1) * self.song_batch)
            y_indexes = range(self.song_batch_index * self.song_batch, (self.song_batch_index + 1) * self.song_batch)

            self.x_indices, self.y_indices = self.transform_data([self.ids[k] for k in x_indexes], [self.labels[k] for k in y_indexes])

            if self.shuffle_data:
                self.x_indices, self.y_indices = shuffle(x_indexes, y_indexes, random_state=0)

            self.song_batch_index += 1

        # Generate batch
        x = np.zeros((self.batch_size, self.dim[0], 1))
        y = np.zeros((self.batch_size, self.n_classes))

        for i in range(0, self.batch_size):
            x[i] = self.x_indices[i + ((self.batch_size*index) % self.song_batch), :]
            y[i] = self.y_indices[i + ((self.batch_size * index) % self.song_batch), :]

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        ids_tmp = self.ids
        labels_tmp = self.labels

        if self.shuffle_data:
            self.ids, self.labels = shuffle(ids_tmp, labels_tmp, random_state=0)

    def __data_generation(self, ids_temp, labels_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        x, y = self.transform_data(ids_temp, labels_temp, self.batch_size)
        return x, y  # keras.utils.to_categorical(y, num_classes=self.n_classes)
