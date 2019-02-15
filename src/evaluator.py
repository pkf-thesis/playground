from models.base_model import BaseModel

class Evaluator:
    
    def evaluate(self, model: BaseModel, x_test, y_test):
        score = model.evaluate(x_test, y_test, verbose=0)
        print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))