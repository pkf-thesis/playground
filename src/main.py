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

batch_size = 25
learning_rates = [0.01, 0.002, 0.0004, 0.00008, 0.000016]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "-data", help="gtzan, mtat or msd")
    parser.add_argument("-logging", help="Logs to csv file")
    parser.add_argument("-gpu", type=list, help="Run on gpu's, and which")
    parser.add_argument("-local", help="Whether to run local or on server")

    args = parser.parse_args()

    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data(args)

    output = open("../results/results.txt", "a")

    'First learning rate'
    if args.local:
        base_model = Basic2DCNN(song_length=640512, dim=(128, 126), n_channels=1, batch_size=batch_size,
                                weight_name='../results/best_weights_%s_%s.hdf5', args=args)
    else:
        base_model = SampleCNN39(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size,
                                 weight_name='../results/best_weights_%s_%s_adam.hdf5', args=args)

    start = time.time()
    output.write("Start Training %s with Adam optimizer - %s \n" % (base_model.model_name, start))
    lr = learning_rates[0]
    model = base_model.train(x_train, y_train, x_valid, y_valid, epoch_size=1000, lr=lr)

    print("Testing")
    x_pred = evaluator.predict(base_model, model, x_test, lr)

    'Save predictions'
    np.save("../results/predictions_%s_%s_%s_adam.npy" % (base_model.model_name, args.d, lr), x_pred)

    test_result = evaluator.mean_roc_auc(x_pred, y_test)
    print("Mean ROC-AUC: %s" % test_result)
    output.write("%lr -  Mean ROC-AUC: %s \n" % (lr, test_result))

    # 'For each learning rate'
    # for lr_index in range(1, len(learning_rates)):
    #     lr = learning_rates[lr_index]
    #     if args.local:
    #         base_model = Basic2DCNN(song_length=640512, dim=(128, 126), n_channels=1, batch_size=batch_size,
    #                                 weight_name='../results/best_weights_%s_%s.hdf5', args=args)
    #     else:
    #         base_model = SampleCNN39(640512, dim=(3 * 3 ** 9,), n_channels=1, batch_size=batch_size,
    #                                  weight_name='../results/best_weights_%s_%s.hdf5', args=args)
    #
    #     print('Train %s' % lr)
    #     model = base_model.retrain(x_train, y_train, x_valid, y_valid, epoch_size=100, lr=lr,
    #                                lr_prev=learning_rates[lr_index-1])
    #
    #     print("Testing")
    #     x_pred = evaluator.predict(base_model, model, x_test, lr)
    #
    #     'Save predictions'
    #     np.save("../results/predictions_%s_%s_%s.npy" % (base_model.model_name, args.d, lr), x_pred)
    #
    #     test_result = evaluator.mean_roc_auc(x_pred, y_test)
    #     print("Mean ROC-AUC: %s" % test_result)
    #     output.write("%lr -  Mean ROC-AUC: %s \n" % (lr, test_result))

    end = time.time()
    output.write("End Training %s - %s \n" % (base_model.model_name, end))

    output.close()
