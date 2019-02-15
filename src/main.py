import numpy
import sys
import mp3_to_npy_convertor
import train_test_divider
import os

from base_model import BaseModel
from evaluator import Evaluator
import sqllite_repository as sql

if not os.path.exists("../npys"):
    mp3_to_npy_convertor.convert_files("../data", "../npys/", 22050, 640512)

(train, test) = train_test_divider.splitData("../npys", 0.8)
(train_x, train_y) = sql.fetchTagsFromSongs(train)

'Initiate model and train'
model = BaseModel()
model.train(train_x, train_y, 10, 50)

'Evaluate model'
evaluator = Evaluator()
(test_x, test_y) = sql.fetchTagsFromSongs(test)
evaluator.evaluate(model, test_x, test_y)
