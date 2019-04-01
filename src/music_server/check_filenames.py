import src.sqllite_repository as sql

# ids_from_db = sql.fetch_all_songs()

base_path = "../../data/%s"

loaded_train_ids = [song.split('/')[-1].rstrip() for song in open(base_path % "mtat/train_path.txt")]
loaded_valid_ids = [song.split('/')[-1].rstrip() for song in open(base_path % "mtat/valid_path.txt")]
loaded_test_ids = [song.split('/')[-1].rstrip() for song in open(base_path % "mtat/test_path.txt")]

train_ids = [song.rstrip() for song in open(base_path % "mtat_ids/train.txt")]
valid_ids = [song.rstrip() for song in open(base_path % "mtat_ids/valid.txt")]
test_ids = [song.rstrip() for song in open(base_path % "mtat_ids/test.txt")]

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
