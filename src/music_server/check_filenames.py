import os
import src.sqllite_repository as sql

ids_from_db = sql.fetch_all_songs()

train_ids = list(open("../../data/msd_ids/train").read().split('\n'))
valid_ids = list(open("../../data/msd_ids/valid").read().split('\n'))
test_ids = list(open("../../data/msd_ids/test").read().split('\n'))

all_list = train_ids + valid_ids + test_ids

count = 0
for id in train_ids:
    if id.rstrip() not in ids_from_db:
        count += 1

print("done train")
for id in valid_ids:
    if id.rstrip() not in ids_from_db:
        count += 1

print("done valid")
for id in test_ids:
    if id.rstrip() not in ids_from_db:
        count += 1

print(count)
