import sys
import numpy as np
from evaluator import plot_confusion_matrix2

predictions = np.load(sys.argv[1])
truths = np.load(sys.argv[2])
labels = [label.rstrip() for label in open("../data/mtat/tags.txt")]

confusion_matrix = plot_confusion_matrix2(predictions, truths, labels, 4)