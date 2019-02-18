import numpy as np
import keras
import os

genres = {'classical':0, 'pop':1, 'blues':2, 'country':3, 'disco': 4, 'hiphop': 5,
          'jazz':6, 'metal':7, 'reggae':8, 'rock':9}

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, labels, batch_size=32, dim=(64000,), n_channels=1,
                 n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        ids_temp = [self.ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(ids_temp):
            # Store sample
            x = np.load('../npys/' + id)
            x = x.reshape((-1, 1))
            X[i, ] = x

            # Store class
            y[i] = genres[self.labels[i]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
