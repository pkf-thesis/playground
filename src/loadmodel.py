import numpy as np
import itertools

from src.models.max_average_net import MaxAverageNet
from src.utils.utils import load_multigpu_checkpoint_weights


def predict(model):
    x_test = [song.rstrip() for song in open("../data/mtat/test_path.txt")]
    sample_length = 59049
    num_segments = 10

    x_test_temp = np.zeros((num_segments, sample_length, 1))
    x_pred = np.zeros((len(x_test), 50))

    for i, song_id in enumerate(x_test):
        song = np.load("../sdb/data/%s/%s.npz" % ("mtat", song_id))['arr_0']

        for segment in range(0, num_segments):
            x_test_temp[segment] = song[segment * sample_length:
                                        segment * sample_length + sample_length].reshape((-1, 1))

        x_pred[i] = np.mean(model.predict(x_test_temp), axis=0)

    return x_pred

"""
def plot_confusion_matrix(predictions, truths, target_names, title='Confusion matrix', cmap=None, normalize=True):
    cm = multilabel_confusion_matrix(truths, predictions)
    print(cm)
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
    plt.savefig("confusion_matrix.png")

#loaded_model = samplecnn.build_model()
#os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
#loaded_model = multi_gpu_model(loaded_model, gpus=2)
# load weights into new model
#loaded_model.load_weights("best_weights_SampleCNN_3_9_(59049,)_8e-05.hdf5")
#print("Loaded model from disk")
#print("Predicting model")
predictions = np.load("prediction.npy")
predictions = (predictions > 0.5).astype(int)
truths = np.load("../data/mtat/y_test_pub.npy").astype(int)
labels = [label.rstrip() for label in open("../data/mtat/tags.txt")]
plot_confusion_matrix(predictions, truths, labels)
"""