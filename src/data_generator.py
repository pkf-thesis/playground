import numpy as np
import keras
from sklearn.utils import shuffle


class DataGenerator(keras.utils.Sequence):

    def __init__(self, transform, ids, labels, batch_size, dim, n_channels, n_classes, shuffle=True):
        self.transform_data = transform
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        x_indexes = range(index*self.batch_size, (index+1)*self.batch_size)
        y_indexes = range(index*self.batch_size, (index+1)*self.batch_size)


        # Find list of IDs
        ids_temp = [self.ids[k] for k in x_indexes]
        labels_temp = [self.labels[k] for k in y_indexes]

        # Generate data
        x, y = self.__data_generation(ids_temp, labels_temp)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        ids_tmp = self.ids
        labels_tmp = self.labels

        if self.shuffle:
            self.ids, self.labels = shuffle(ids_tmp, labels_tmp, random_state=0)

    def __data_generation(self, ids_temp, labels_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        x, y = self.transform_data(ids_temp, labels_temp, self.batch_size)
        return x, y  # keras.utils.to_categorical(y, num_classes=self.n_classes)
