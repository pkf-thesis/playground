from typing import List, Tuple

import argparse
import os
#import keras
#import tensorflow as tf

import music_to_npy_convertor, train_test_divider
from models.basic_2d_cnn import Basic2DCNN
from models.sample_cnn_3_9 import SampleCNN39
from evaluator import Evaluator
import utils.gtzan_genres as gtzan
import sqllite_repository as sql

def get_data(args) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split data into train and test"""

    x_train, y_train, x_test, y_test = None, None, None, None

    if args.d == 'gtzan':
        x_train, y_train, x_test, y_test = train_test_divider.split_data_sklearn("../npys", 0.2)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "-data", help="gtzan, mtat or msd")
    parser.add_argument("-logging", help="Logs to csv file")
    parser.add_argument("-gpu", type=list, help="Run on gpu's, and which")
    parser.add_argument("-local", help="Whether to run local or on server")

    args = parser.parse_args()

    if not os.path.exists("../npys"):
        music_to_npy_convertor.convert_files("../data/gtzan/", "../npys/", 22050, 640512)

    x_train, y_train, x_test, y_test = get_data(args)

    'Initiate model'
    if args.local:
        base_model = Basic2DCNN(song_length=int(640512 * 0.1), dim=(128, 126), n_channels=1, n_labels=10, args=args)
    else:
        base_model = SampleCNN39(640512, dim=(3 * 3**9,), n_channels=1, n_labels=10, args=args)

    if not os.path.exists(args.logging):
        os.makedirs(os.path.dirname(args.logging + base_model.model_name + '.csv'))

    learning_rates = [0.01, 0.002, 0.0004, 0.00008, 0.000016]

    for lr in learning_rates:

        'Train'
        base_model.train(x_train, y_train, epoch_size=100, lr=lr, batch_size=10)

        #Load best model

        'Evaluate model'
        evaluator = Evaluator()
        # (test_x, test_y) = sql.fetchTagsFromSongs(test)
        evaluator.evaluate(base_model, x_test, y_test)
