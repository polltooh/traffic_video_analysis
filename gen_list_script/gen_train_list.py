import os
import random
import file_io

if __name__ == "__main__":
    file_list_dir = "../file_list/"
    data_ext = "mixed10"
    label_ext = "desmap"
    file_dir = "../data/Cam253/[Cam253]-2016_4_21_15h_150f/"

    data_list = file_io.get_listfile(file_dir, data_ext)

    partition = 0.7
    train_data_len = int(len(data_list) * partition)

    random.shuffle(data_list)
    train_data = data_list[:train_data_len]
    test_data = data_list[train_data_len:]

    train_list = [d + " " + d.replace(data_ext, label_ext) for d in train_data]
    test_list = [d + " " + d.replace(data_ext, label_ext) for d in test_data]

    train_file_list_name = 'train_list1.txt'
    file_io.save_file(train_list, file_list_dir + train_file_list_name, True)

    test_file_list_name = 'test_list1.txt'
    file_io.save_file(test_list, file_list_dir + test_file_list_name, True)


