from typing import List, Tuple

import argparse
import os
import keras
import tensorflow as tf

import music_to_npy_convertor, train_test_divider
from models.basic_1d_cnn import Basic1DCNN
from models.basic_2d_cnn import Basic2DCNN
from models.sample_cnn_3_9 import SampleCNN39
from evaluator import Evaluator
import utils.gtzan_genres as gtzan
import sqllite_repository as sql

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

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

    args = parser.parse_args()

    if not os.path.exists("../npys"):
        music_to_npy_convertor.convert_files("../data/gtzan/", "../npys/", 22050, 640512)

    x_train, y_train, x_test, y_test = get_data(args)

    'Initiate model'
    #base_model = Simple1DCNN(640512, dim=(640512,), n_channels=1, n_labels=10, logging=args.logging)
    base_model = SampleCNN39(640512, dim=(3 * 3**9,), n_channels=1, n_labels=10, logging=args.logging)
    #base_model = Simple2DCNN(song_length=int(640512 * 0.1), dim=(128, 126), n_channels=1, n_labels=10, logging=args.logging)

    if not os.path.exists(args.logging):
        os.makedirs(os.path.dirname(args.logging + base_model.model_name + '.csv'))

    'Train'
    base_model.train(x_train, y_train, epoch_size=5, batch_size=25)

    'Evaluate model'
    evaluator = Evaluator()
    # (test_x, test_y) = sql.fetchTagsFromSongs(test)
    evaluator.evaluate(base_model, x_test, y_test)
