from typing import List

from models.base_model import BaseModel
from data_generator import DataGenerator


class Evaluator:
    
    def evaluate(self, base_model: BaseModel, x_test: List[str], y_test: List[str]) -> None:
        test_generator = DataGenerator(base_model.transform_data, x_test, y_test, batch_size=5,
                                  dim=base_model.dimension, n_classes=base_model.n_labels)

        score = base_model.model.evaluate_generator(test_generator, len(x_test) / 5, verbose=0)
        print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))
