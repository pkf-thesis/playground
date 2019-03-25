import os

import numpy as np
import librosa

def convert_files(path, feature_path, frequency, max_length):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)
            dirname = int(count/100)
            count = count + 1
            if file.endswith(".au"):
                save_name = feature_path + str(dirname) + "/" + file.replace('.au', '')
            
                if not os.path.exists(os.path.dirname(save_name)):
                    print(os.path.dirname(save_name))
                    os.makedirs(os.path.dirname(save_name))

                if os.path.isfile(save_name) == 1:
                    print(save_name + '_file_exist!!!!!!!!!!!!!!!')
                    continue
                print(file_name)
                try:
                    y, sr = librosa.load(file_name)
                except:
                    continue
                y = y.astype(np.float32)

                if len(y) > max_length:
                    y = y[0:max_length]

                print(len(y), save_name)
                np.savez_compressed(save_name, y)

convert_files("data/gtzan/", "npys/gtzan/", 22050, 640512)
