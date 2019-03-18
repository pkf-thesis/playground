import os

import numpy as np
import librosa
import sqllite_repository as sql


def convert_files(path, feature_path, frequency, max_length):
    songs = sql.fetch_all_songs()
    dir_name = ''

    for root, dirs, files in os.walk(path):
        for i, file in enumerate(files):
            splitted_file = file.split('.')
            if splitted_file[0] in songs:
                file_name = os.path.join(root, file)
                if i % 1000 == 0:
                    dir_name = "%s - %s" % (i, i + 1000)
                if file.endswith(".mp3"):
                    save_name = feature_path + str(dir_name) + "/" + file.replace('.mp3', '')

                    if not os.path.exists(os.path.dirname(save_name)):
                        print(os.path.dirname(save_name))
                        os.makedirs(os.path.dirname(save_name))

                    if os.path.isfile(save_name) == 1:
                        print(save_name + '_file_exist!!!!!!!!!!!!!!!')
                        continue
                    print(file_name)
                    try:
                        y, sr = librosa.load(file_name, sr=frequency)
                    except:
                        print("Error loading: %s" % save_name)
                        continue

                    y = y.astype(np.float32)

                    if len(y) > max_length:
                        y = y[0:max_length]

                    print(len(y), save_name)
                    np.save(save_name, y)


convert_files("/30T/Music/MSD/audio", "npys/", 22050, 640512)

