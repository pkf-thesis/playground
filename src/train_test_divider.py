import os
from typing import List, Tuple
import utils.gtzan_genres as gtzan

from sklearn.model_selection import train_test_split


def split_data_sklearn(path: str, test_size: float) -> Tuple[List[str], List[str], List[str], List[str]]:
    ids = [song.rstrip() for song in open(path)]

    train_input = list(map(lambda song_id: gtzan.genres[song_id.split(".")[0].split("/")[1]], ids))
    x_train, x_test, y_train, y_test = train_test_split(ids, train_input, test_size=test_size)

    return x_train, y_train, x_test, y_test


def split_data(path: str, train_ratio: float) -> Tuple[List[str], List[str], List[str], List[str]]:
    ids = os.listdir(path)
    train_index = int(len(ids) * train_ratio)
    train_x = ids[:train_index]
    test_x = ids[train_index:]
    train_y = list(map(lambda id: id.split(".")[0], train_x))
    test_y = list(map(lambda id: id.split(".")[0], test_x))

    return train_x, train_y, test_x, test_y