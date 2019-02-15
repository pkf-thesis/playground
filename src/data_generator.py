import numpy as np
import keras
import os

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ids, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        self.dim = np.load("../npys/"+os.listdir("../npys")[0]).shape
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
            X[i,] = np.load('../npys/' + id + '.npy')

            # Store class
            y[i] = self.labels[id]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
