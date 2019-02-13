import numpy
import sys
import mp3_to_npy_convertor
import train_test_divider
import os

if not os.path.exists("../npys"):
    mp3_to_npy_convertor.convert_files("../data", "../npys/", 22050, 640512)

(train, test) = train_test_divider.splitData("../npys", 0.8)
(ids, labels) = sql.load(train)

'Initiate model with train data'
model = Model()
model.train(ids, labels)

'Evaluate model'
(testIds, testLabels) = sql.load(test)
evaluator.evaluate(model, testIds, testLabels)
