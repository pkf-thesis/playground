import numpy
import sys
import music_to_npy_convertor
import train_test_divider
import os

from models.simple_2d_cnn import Simple2DCNN
from evaluator import Evaluator
import sqllite_repository as sql

if not os.path.exists("../npys"):
    music_to_npy_convertor.convert_files("../data/gtzan/", "../npys/", 22050, 640512)

(train_x, test) = train_test_divider.splitData("../npys", 1)
train_y = list(set(map(lambda id: id.split(".")[0], train_x)))
shape = numpy.load("../npys/"+os.listdir("../npys")[0]).shape
print(shape)

'Initiate model and train'
model = Simple2DCNN(shape,len(train_y))
model.train(train_x, train_y, 10, len(train_y))

'Evaluate model'
evaluator = Evaluator()
(test_x, test_y) = sql.fetchTagsFromSongs(test)
evaluator.evaluate(model, test_x, test_y)
