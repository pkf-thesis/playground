from typing import List, Tuple
import numpy as np

from models.base_model import BaseModel


class SampleCNN39(BaseModel):

    model_name = "SampleCNN_3_9"

    def transform_data(self, ids_temp: List[str], batch_size: int) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
