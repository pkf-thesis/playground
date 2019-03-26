from typing import List

from models.base_model import BaseModel
from data_generator import DataGenerator

import numpy as np


class Evaluator:

    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def evaluate(self, base_model: BaseModel, model, x_test: List[str], y_test: List[str]) -> None:

        test_generator = DataGenerator(base_model.transform_data, x_test, y_test, batch_size=5,
                                  dim=base_model.dimension, n_classes=base_model.n_labels)

        score = model.evaluate_generator(test_generator, len(x_test) / 5)
        print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

        # Testing predict
        # song = np.load('../npys/blues.00015.npy')
        # song = song[0:3 * 3 ** 9]
        # song = song.reshape((-1, 1))
        # print(song.shape)
        # song = song.reshape(1, 3 * 3**9, 1)
        # prediction = model.predict(song)
        # print(prediction)
