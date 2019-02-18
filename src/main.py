import numpy
import os

import music_to_npy_convertor, train_test_divider
from models.simple_1d_cnn import Simple1DCNN
from models.simple_2d_cnn import Simple2DCNN
from evaluator import Evaluator
import sqllite_repository as sql

if not os.path.exists("../npys"):
    music_to_npy_convertor.convert_files("../data/gtzan_small/", "../npys/", 22050, 64000)

'Split data into train and test'
(train_x, train_y, test_x, test_y) = train_test_divider.splitData("../npys", 0.8)

shape = numpy.load("../npys/"+os.listdir("../npys")[0]).shape
print(shape)

'Initiate model and train'
model = Simple1DCNN((64000, 1), 10)
model.train(train_x, train_y, 10, 10)

'Evaluate model'
evaluator = Evaluator()
evaluator.evaluate(model, test_x, test_y)
