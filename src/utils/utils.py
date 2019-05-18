import os
from queue import Queue
from sys import stderr

import h5py
import numpy as np

import train_test_divider as train_test_divider

def make_path(*paths):
    path = os.path.join(*[str(path) for path in paths])
    path = os.path.realpath(path)
    return path


def calculate_num_segments(sample_length):
    return 640512 // sample_length


def train_generator(train_list, y_train_init, batch_size, song_batch, sample_length, num_tags, dataset, path):
    i = 0
    j = 0

    batch_size = batch_size
    train_length = len(train_list)
    subset_size = int(train_length / song_batch)
    # MSD: example, total 201680, song_batch=40, subset_size=5042, batch_size=50
    # MTAT: total 15244, song_batch=37, sub_set_size=412, batch_size=25

    num_segments = calculate_num_segments(sample_length)

    while 1:
        # load subset
        x_train_sub = np.zeros((subset_size * num_segments, sample_length, 1))
        y_train_sub = np.zeros((subset_size, num_tags))

        for subset_size_index in range(0, subset_size):
            '''	
            # for debugging
            if iter == 0:
                print '\n'+str(subset_size_index * subset_size + i)
            else:
                print subset_size_index * subset_size + i, subset_size_index, i
            '''

            '''
            If we are trying to load a song which is outside our list, 
            this can happen if song_batch does not add up to the length of our train samples
            '''
            try:
                # load x_train
                tmp = np.load(path % (dataset, train_list[subset_size_index * song_batch + i]))['arr_0']
            except:
                break

            for num_segments_index in range(0, num_segments):
                x_train_sub[num_segments * subset_size_index + num_segments_index, :, 0] = \
                    tmp[num_segments_index * sample_length:num_segments_index * sample_length + sample_length]

            y_train_sub[subset_size_index] = y_train_init[subset_size_index * song_batch + i, :]

        # Duplication
        y_train_sub = np.repeat(y_train_sub, num_segments, axis=0)
        # print 'sub train set loaded!' + str(i)

        # segments randomization
        tmp_train = np.arange(num_segments * subset_size)
        np.random.shuffle(tmp_train)
        x_train_sub = x_train_sub[tmp_train]
        y_train_sub = y_train_sub[tmp_train]

        # segment flatten
        x_train_sub_batch = np.zeros((batch_size, sample_length, 1))
        y_train_sub_batch = np.zeros((batch_size, num_tags))

        for iter2 in range(0, int(subset_size * num_segments / batch_size)):

            # batch set
            for batch_index in range(0, batch_size):
                x_train_sub_batch[batch_index] = x_train_sub[
                                                 int(batch_index * subset_size * num_segments / batch_size) + j, :]
                y_train_sub_batch[batch_index] = y_train_sub[
                                                 int(batch_index * subset_size * num_segments / batch_size) + j, :]

                """
                # for debugging
                if batch_index == 0:
                    print('\n'+str(batch_index * subset_size * num_segments / batch_size + j))
                else:
                    print(batch_index * subset_size * num_segments / batch_size + j)
                """

            j = j + 1
            yield (x_train_sub_batch, y_train_sub_batch)

        if j == int(subset_size * num_segments / batch_size):
            j = 0
        i = i + 1
        if i == song_batch:
            i = 0

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


def load_multigpu_checkpoint_weights(model, h5py_file):
    """
    Loads the weights of a weight checkpoint from a multi-gpu
    keras model.

    Input:

        model - keras model to load weights into

        h5py_file - path to the h5py weights file

    Output:
        None
    """

    print("Setting weights...")
    with h5py.File(h5py_file, "r") as file:
        model_name = None
        for key in file.keys():
            if 'model' in key:
                model_name = key

        # Get model subset in file - other layers are empty
        weight_file = file[model_name]

        for layer in model.layers:

            try:
                layer_weights = weight_file[layer.name]

            except:
                # No weights saved for layer
                continue

            try:
                weights = []
                # Extract weights
                for term in layer_weights:
                    if isinstance(layer_weights[term], h5py.Dataset):
                        # Convert weights to numpy array and prepend to list
                        weights.insert(0, np.array(layer_weights[term]))

                # Load weights to model
                layer.set_weights(weights)

            except Exception as e:
                print("Error: Could not load weights for layer:", layer.name, file=stderr)


def check_weights(build_model, file):
    load_multigpu_checkpoint_weights(build_model, file)
    weights = build_model.layers[7].get_weights()
    weights2 = build_model.layers[11].get_weights()
    weights3 = build_model.layers[15].get_weights()
    weights4 = build_model.layers[19].get_weights()
    weights5 = build_model.layers[23].get_weights()
    weights6 = build_model.layers[27].get_weights()
    weights7 = build_model.layers[31].get_weights()
    weights8 = build_model.layers[35].get_weights()
    weights9 = build_model.layers[39].get_weights()

    print("%s, %s, %s, %s, %s, %s, %s, %s, %s" % (weights, weights2, weights3, weights4, weights5,
                                                  weights6, weights7, weights8, weights9))


def check_weight(build_model, file):
    load_multigpu_checkpoint_weights(build_model, file)
    weights = build_model.layers[7].get_weights()

    print("%s" % (weights))



