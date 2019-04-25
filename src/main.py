import argparse
import numpy as np
import time

import music_to_npy_convertor, train_test_divider as train_test_divider
from models.basic_2d_cnn import Basic2DCNN
from models.sample_cnn_3_9 import SampleCNN39
from models.sample_cnn_3_9_resnet import SampleCNN39ResNet
from models.sample_cnn_deep_resnet import SampleCNNDeepResNet
import evaluator as evaluator
from utils.utils import get_data

import experiments as exp

batch_size = 25


def build_basic():
    return Basic2DCNN(song_length=640512, dim=(128, 126), n_channels=1, batch_size=batch_size,
                      weight_name='../results/best_weights_%s_%s.hdf5', args=args)


def build_sample_39():
    return SampleCNN39(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "-data", help="gtzan, mtat or msd")
    parser.add_argument("-logging", help="Logs to csv file")
    parser.add_argument("-gpu", type=list, help="Run on gpu's, and which")
    parser.add_argument("-local", help="Whether to run local or on server")

    args = parser.parse_args()

    build_model = None
    if args.local:
        build_model = build_basic
    else:
        build_model = build_sample_39

    #exp.run_experiment(build_model, args)
    exp.run_cross_experiment(build_model, args)
