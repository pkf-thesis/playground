import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import time

import evaluator as evaluator
from utils.utils import get_data

learning_rates = [0.01, 0.002, 0.0004, 0.00008, 0.000016]


def run_experiment(build_model, args):
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data(args)

    output = open("../results/results.txt", "a")

    'First learning rate'
    base_model = build_model()

    start = time.time()
    output.write("Start Training %s - %s \n" % (base_model.model_name, start))
    print('Train first learning rate')
    lr = learning_rates[0]
    weight_name = '../results/best_weights_%s_%s.hdf5' % (base_model.model_name, lr)
    model = base_model.train(x_train, y_train, x_valid, y_valid, epoch_size=100, lr=lr, weight_name=weight_name)

    print("Testing")
    model.load_weights(weight_name)
    x_pred = evaluator.predict(base_model, model, x_test)

    'Save predictions'
    np.save("../results/predictions_%s_%s_%s.npy" % (base_model.model_name, args.d, lr), x_pred)

    test_result = evaluator.mean_roc_auc(x_pred, y_test)
    print("Mean ROC-AUC: %s" % test_result)
    output.write("%lr -  Mean ROC-AUC: %s \n" % (lr, test_result))

    'For each learning rate'
    for lr_index in range(1, len(learning_rates)):
        lr = learning_rates[lr_index]

        base_model = build_model()

        print('Train %s' % lr)
        weight_name = '../results/best_weights_%s_%s.hdf5' % (base_model.model_name, lr)
        model = base_model.retrain(x_train, y_train, x_valid, y_valid, epoch_size=100, lr=lr,
                                   lr_prev=learning_rates[lr_index - 1], weight_name=weight_name)

        print("Testing")
        model.load_weights(weight_name)
        x_pred = evaluator.predict(base_model, model, x_test)

        'Save predictions'
        np.save("../results/predictions_%s_%s_%s.npy" % (base_model.model_name, args.d, lr), x_pred)

        test_result = evaluator.mean_roc_auc(x_pred, y_test)
        print("Mean ROC-AUC: %s" % test_result)
        output.write("%lr -  Mean ROC-AUC: %s \n" % (lr, test_result))

    end = time.time()
    output.write("End Training %s - %s" % (base_model.model_name, end))

    output.close()


def run_cross_experiment(build_model, args):
    base_path = "../data/mtat/cross/%s_%s"

    for i in range(0, 5):
        train_ids = [song.rstrip() for song in open(base_path % (i, "train.txt"))]
        Y_train = np.load(base_path % (i, "train.npy"))
        x_test = [song.rstrip() for song in open(base_path % (i, "test.txt"))]
        y_test = np.load((base_path % (i, "test.npy")))

        # Split into train and validation set
        mskf = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        for train_idx, valid_idx in mskf.split(train_ids, Y_train):
            x_train = []
            y_train = Y_train[train_idx]
            for idx in train_idx:
                x_train.append(train_ids[idx])

            x_valid = []
            y_valid = Y_train[valid_idx]
            for idx in valid_idx:
                x_valid.append(train_ids[idx])

        output = open("../results/cross/results.txt", "a")

        'First learning rate'
        base_model = build_model()

        start = time.time()
        output.write("Start Cross %s Training %s - %s \n" % (i, base_model.model_name, start))
        print('Train first learning rate')
        lr = learning_rates[0]
        weight_name = '../results/cross/%s_best_weights_%s_%s.hdf5' % (i, base_model.model_name, lr)
        model = base_model.train(x_train, y_train, x_valid, y_valid, epoch_size=100, lr=lr, weight_name=weight_name)

        print("Testing")
        model.load_weights(weight_name)
        x_pred = evaluator.predict(base_model, model, x_test)

        'Save predictions'
        np.save("../results/cross/predictions_%s_%s_%s.npy" % (base_model.model_name, args.d, lr), x_pred)

        test_result = evaluator.mean_roc_auc(x_pred, y_test)
        print("Mean ROC-AUC: %s" % test_result)
        output.write("%lr -  Mean ROC-AUC: %s \n" % (lr, test_result))

        'For each learning rate'
        for lr_index in range(1, len(learning_rates)):
            lr = learning_rates[lr_index]

            base_model = build_model()

            print('Train %s' % lr)
            weight_name = '../results/cross/%s_best_weights_%s_%s.hdf5' % (i, base_model.model_name, lr)
            model = base_model.retrain(x_train, y_train, x_valid, y_valid, epoch_size=100, lr=lr,
                                       lr_prev=learning_rates[lr_index - 1], weight_name=weight_name)

            print("Testing")
            model.load_weights(weight_name)
            x_pred = evaluator.predict(base_model, model, x_test)

            'Save predictions'
            np.save("../results/cross/predictions_%s_%s_%s.npy" % (base_model.model_name, args.d, lr), x_pred)

            test_result = evaluator.mean_roc_auc(x_pred, y_test)
            print("Mean ROC-AUC: %s" % test_result)
            output.write("%lr -  Mean ROC-AUC: %s \n" % (lr, test_result))

        end = time.time()
        output.write("End Training %s - %s" % (base_model.model_name, end))

        output.close()
