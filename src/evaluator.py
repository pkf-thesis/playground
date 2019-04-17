from typing import List

from sklearn.metrics import roc_auc_score

import numpy as np

from data_generator import DataGenerator
from utils import utils


def evaluate(base_model, model, x_test: List[str], y_test: List[str]) -> None:
    test_generator = DataGenerator(base_model.transform_data, x_test, y_test, batch_size=25,
                                   dim=base_model.dimension, n_channels=base_model.n_channels,
                                   n_classes=base_model.n_labels)

    score = model.evaluate_generator(test_generator, len(x_test) / 5)
    print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))


def predict(base_model, model, x_test: List[str], lr):
    """Load best weights"""
    if lr is not None:
        model.load_weights(base_model.weight_name % (base_model.model_name, lr))

    sample_length = base_model.dimension[0]
    num_segments = utils.calculate_num_segments(sample_length)

    x_test_temp = np.zeros((num_segments, sample_length, 1))
    x_pred = np.zeros((len(x_test), base_model.n_labels))

    for i, song_id in enumerate(x_test):
        song = np.load(base_model.path % (base_model.dataset, song_id))['arr_0']

        for segment in range(0, num_segments):
            x_test_temp[segment] = song[segment * sample_length:
                                        segment * sample_length + sample_length].reshape((-1, 1))

        x_pred[i] = np.mean(model.predict(x_test_temp), axis=0)

    return x_pred


# Example
# predictions   = array([[0.54, 0.98, 0.43], [0.32, 0.18, 0.78], [0.78, 0.76, 0.86]])
# truths        = array([[1, 1, 0], [0, 0, 1], [1, 1, 0]])
# mean_roc_auc  = 0.66
def mean_roc_auc(predictions, truths):
    num_predictions = len(predictions)
    n_labels = len(truths[0])
    auc = np.zeros(n_labels)
    label_truths = np.zeros((n_labels, num_predictions))
    label_predictions = np.zeros((n_labels, num_predictions))
    for labelIndex in range(n_labels):
        for index in range(num_predictions):
            label_predictions[labelIndex, index] = predictions[index, labelIndex]
            label_truths[labelIndex, index] = truths[index, labelIndex]
    for i in range(n_labels):
        truths = label_truths[i]
        predictions = label_predictions[i]
        auc[i] = roc_auc_score(truths, predictions)
    return np.mean(auc)


    """
    def plot_confusion_matrix(predictions, truths, target_names, title='Confusion matrix', cmap=None, normalize=True):
        cm = confusion_matrix(truths, predictions)
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if cmap is None:
            cmap = plt.get_cmap('Blues')
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig("confusion_matrix.png", bbox_inches="tight")
        """
