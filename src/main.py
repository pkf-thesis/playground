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
(ids, labels) = sql.load(train)

'Initiate model and train'
model = BaseModel()
model.train(ids, labels, 10, 50)

'Evaluate model'
evaluator = Evaluator()
(testIds, testLabels) = sql.load(test)
evaluator.evaluate(model, testIds, testLabels)
