from typing import List, Tuple

import argparse
import os
import numpy as np

import music_to_npy_convertor, train_test_divider
from models.basic_2d_cnn import Basic2DCNN
from models.sample_cnn_3_9 import SampleCNN39
from evaluator import Evaluator

batch_size = 20
learning_rates = [0.01, 0.002, 0.0004, 0.00008, 0.000016]


def get_data(args):
    """Split data into train and test"""

    x_train, y_train, x_valid, y_valid, x_test, y_test = None, None, None, None, None, None

    if args.d == 'gtzan':
        validation_size = 0.1

        x_train, y_train, x_test, y_test = train_test_divider.split_data_sklearn("../data/gtzan/ids.txt", 0.2)

        num_train = len(x_train)
        x_valid = x_train[:int(num_train * validation_size)]
        y_valid = y_train[:int(num_train * validation_size)]
        x_train = x_train[int(num_train * validation_size):]
        y_train = y_train[int(num_train * validation_size):]

    elif args.d == 'msd':
        base_path = "../data/msd/"
        x_train = [song.rstrip() for song in open(base_path + "train_path.txt")]
        y_train = np.load(base_path + "y_train.npz")['arr_0']

        # Fix for removing npz files which can't be loaded
        error_idx = x_train.index("292000-293000/TRCTUYS128F425175B")
        del x_train[error_idx]
        y_train = np.delete(y_train, [error_idx], 0)

        x_valid = [song.rstrip() for song in open(base_path + "valid_path.txt")]
        y_valid = np.load(base_path + "y_valid.npz")['arr_0']

        x_test = [song.rstrip() for song in open(base_path + "test_path.txt")]
        y_test = np.load(base_path + "y_test.npz")['arr_0']

    elif args.d == 'mtat':
        base_path = "../data/mtat/"
        x_train = [song.rstrip() for song in open(base_path + "train_path.txt")]
        y_train = np.load(base_path + "y_train_pub.npy")

        x_valid = [song.rstrip() for song in open(base_path + "valid_path.txt")]
        y_valid = np.load(base_path + "y_valid_pub.npy")

        x_test = [song.rstrip() for song in open(base_path + "test_path.txt")]
        y_test = np.load(base_path + "y_test_pub.npy")

    return x_train, y_train, x_valid, y_valid, x_test, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "-data", help="gtzan, mtat or msd")
    parser.add_argument("-logging", help="Logs to csv file")
    parser.add_argument("-gpu", type=list, help="Run on gpu's, and which")
    parser.add_argument("-local", help="Whether to run local or on server")

    args = parser.parse_args()

    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data(args)

    'Initiate model'
    if args.local:
        base_model = Basic2DCNN(song_length=640512, dim=(128, 126), n_channels=1, batch_size=batch_size, args=args)
    else:
        base_model = SampleCNN39(640512, dim=(3 * 3**9,), n_channels=1, batch_size=batch_size, args=args)

    if not os.path.exists(args.logging):
        os.makedirs(os.path.dirname(args.logging + base_model.model_name + '.csv'))

    evaluator = Evaluator(batch_size=batch_size)

    for lr in learning_rates:

        'Train'
        model = base_model.train(x_train, y_train, x_valid, y_valid, epoch_size=100, lr=lr)

        weight_name = 'best_weights_%s_%s_%s.hdf5' % (base_model.model_name, base_model.dimension, lr)
        model.load_weights(weight_name)

        'Evaluate model'
        # (test_x, test_y) = sql.fetchTagsFromSongs(test)
        evaluator.evaluate(base_model, model, x_test, y_test)

