import os

import numpy as np
import librosa

path = "/30T/Music/MSD/audio"
feature_path = "../npys/msd/"
max_length = 640512
frequency = 22050

error_logs = open("../logs.txt", "w")

count = 0


def convert_files():
    print("Converting train data")
    train_ids = [song.rstrip() for song in open("../data/msd/train")]
    convert_list(train_ids, "../data/msd/train_path.txt")

    print("Converting valid data")
    valid_ids = [song.rstrip() for song in open("../data/msd/valid")]
    convert_list(valid_ids, "../data/msd/valid_path.txt")

    print("Converting test data")
    test_ids = [song.rstrip() for song in open("../../data/msd/test")]
    convert_list(test_ids, "../data/msd/test_path.txt")

    error_logs.close()


def convert_list(list_ids, file_name):
    dir_name = ''
    file_ids = open(file_name, "w")
    for root, dirs, files in os.walk(path):
        for file in files:
            splitted_file = file.split('.')
            if splitted_file[0] in list_ids:
                file_name = os.path.join(root, file)
                if count % 1000 == 0:
                    dir_name = "%s - %s" % (count, count + 1000)
                if file.endswith(".mp3"):
                    save_name = feature_path + str(dir_name) + "/" + file.replace('.mp3', '')
                    if not os.path.exists(os.path.dirname(save_name)):
                        os.makedirs(os.path.dirname(save_name))

                    if os.path.isfile(save_name) == 1:
                        continue
                    try:
                        y, sr = librosa.load(file_name, sr=frequency)
                    except:
                        error_logs.write("Error loading: %s\n" % save_name)
                        print("Error loading: %s" % save_name)
                        continue

                    y = y.astype(np.float32)

                    if len(y) > max_length:
                        y = y[0:max_length]

                    file_ids.write("%s\n" % save_name)
                    np.savez_compressed(save_name, y)

    file_ids.close()


convert_files()


