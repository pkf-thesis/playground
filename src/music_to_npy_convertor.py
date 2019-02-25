import os

import numpy as np
import librosa


def convert_files(path: str, feature_path: str, frequency: int, max_length: int) -> None:
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)

            if file.endswith(".au"):  
                save_name = feature_path + file.replace('.au', '')
            
                if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))

                if os.path.isfile(save_name) == 1:
                    print(save_name + '_file_exist!!!!!!!!!!!!!!!')
                    continue
                print(file_name)
                y, sr = librosa.load(file_name)
                y = y.astype(np.float32)

                if len(y) > max_length:
                    y = y[0:max_length]

                print(len(y), save_name)
                np.save(save_name, y)