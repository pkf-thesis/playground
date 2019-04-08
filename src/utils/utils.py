import os
import numpy as np


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
                print '\n'+str(iter*options.partition+i)
            else:
                print iter*options.partition+i,iter,i
            '''

            # load x_train
            tmp = np.load(path % (dataset, train_list[subset_size_index * song_batch + i]))['arr_0']

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

                '''	
                # for debugging
                if iter3 == 0:
                    print '\n'+str(iter3*subset_size*num_segment/batch_size+j)
                else:
                    print iter3*subset_size*num_segment/batch_size+j
                '''

            j = j + 1
            yield (x_train_sub_batch, y_train_sub_batch)

        if j == subset_size * num_segments / batch_size:
            j = 0
        i = i + 1
        if i == song_batch:
            i = 0
