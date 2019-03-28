import numpy as np
import src.sqllite_repository as sql
import src.utils.msd_tags as msd


def convert_tags_to_npy(ids, npy_path):
    y = np.empty((len(ids), len(msd.TAGS)), dtype=bool)
    tid_tag = sql.fetch_tags_from_songs_above_treshold(ids, 50)
    if len(tid_tag) != len(ids):
        print("Skipped dataset %s" % npy_path)
        return
    for i, song in enumerate(ids):
        tags = []
        for tag in msd.TAGS:
            if tag in tid_tag[song]:
                tags.append(True)
            else:
                tags.append(False)
        y[i] = tags
    np.savez_compressed(npy_path, y)


train_ids = [song.split('/')[-1].rstrip() for song in open("../../data/msd/train_path.txt")]
valid_ids = [song.split('/')[-1].rstrip() for song in open("../../data/msd/valid_path.txt")]
test_ids = [song.split('/')[-1].rstrip() for song in open("../../data/msd/test_path.txt")]

convert_tags_to_npy(train_ids, "../../data/msd/y_train")
convert_tags_to_npy(valid_ids, "../../data/msd/y_valid")
convert_tags_to_npy(test_ids, "../../data/msd/y_test")
