import src.sqllite_repository as sql

# ids_from_db = sql.fetch_all_songs()

loaded_train_ids = [song.split('/')[-1].rstrip() for song in open("../../data/msd/train_path.txt")]
loaded_valid_ids = [song.split('/')[-1].rstrip() for song in open("../../data/msd/valid_path.txt")]
loaded_test_ids = [song.split('/')[-1].rstrip() for song in open("../../data/msd/test_path.txt")]

train_ids = [song.rstrip() for song in open("../../data/msd_ids/train")]
valid_ids = list(open("../../data/msd_ids/valid").read().split('\n'))
test_ids = list(open("../../data/msd_ids/test").read().split('\n'))

count = 0

print("Train ids")
diff_train = list(set(train_ids) - set(loaded_train_ids))
for id in diff_train:
    count += 1
    print(id)
print("\n")

print("Valid ids")
diff_valid = list(set(valid_ids) - set(loaded_valid_ids))
for id in diff_valid:
    count += 1
    print(id)
print("\n")

print("Test ids")
diff_test = list(set(test_ids) - set(loaded_test_ids))
for id in diff_test:
    count += 1
    print(id)
print("\n")

print(count)
# all_list = (train_ids + valid_ids + test_ids)
# count = 0
# for id in loaded_train_ids:
#     if id.rstrip() not in all_list:
#         count += 1
#         print(id)
# print("done train\n")
#
# for id in loaded_valid_ids:
#     if id.rstrip() not in all_list:
#         count += 1
#         print(id)
# print("done valid\n")
#
#
# for id in loaded_valid_ids:
#     if id.rstrip() not in all_list:
#         print(id)
#         count += 1
# print("done test\n")
#
# print(count)
