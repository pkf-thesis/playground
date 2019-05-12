import argparse

import music_to_npy_convertor, train_test_divider as train_test_divider
from models.basic_2d_cnn import Basic2DCNN
from models.sample_cnn_3_9 import SampleCNN39
from models.sample_cnn_3_9_resnet import SampleCNN39ResNet
from models.sample_cnn_deep_resnet import SampleCNNDeepResNet
from models.sample_cnn_3_9_max_average import SampleCNNMaxAverage
from models.sample_cnn_avg import SampleCNNAvg
from models.resnet import ResNet
from models.sample_cnn_lstm import SampleCNNLSTM
from models.max_average_net import MaxAverageNet

import experiments as exp

batch_size = 25


def build_basic():
    return Basic2DCNN(song_length=640512, dim=(128, 126), n_channels=1, batch_size=batch_size, args=args)


def build_sample_39():
    return SampleCNN39(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size, args=args)


def build_sample_deep_resnet():
    return SampleCNNDeepResNet(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size, args=args)


def build_sample_max_avg():
    return SampleCNNMaxAverage(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size, args=args)


def build_sample_avg():
    return SampleCNNAvg(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size, args=args)


def build_res_net():
    return ResNet(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size, args=args)


def build_sample_lstm():
    return SampleCNNLSTM(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size, args=args)


def build_max_average_net():
    return MaxAverageNet(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "-data", help="gtzan, mtat or msd")
    parser.add_argument("-logging", help="Logs to csv file")
    parser.add_argument("-gpu", type=list, help="Run on gpu's, and which")
    parser.add_argument("-local", help="Whether to run local or on server")
    parser.add_argument("-cross", help="Whether to run cross experiments or not")

    args = parser.parse_args()

    build_model = None
    if args.local == 'True':
        build_model = build_basic
    else:
        build_model = build_max_average_net()

    if args.cross:
        exp.run_cross_experiment(build_model, args)
    else:
        exp.run_experiment(build_model, args)

