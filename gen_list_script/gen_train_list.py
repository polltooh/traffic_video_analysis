import os
import random
import file_io

if __name__ == "__main__":
    file_list_dir = "../file_list/"
    data_ext = "_resize.resnet_hypercolumn"
    label_ext = "_resize227.desmap"

    
    file_dir = "../data/"
    cam_dir_list = file_io.get_dir_list(file_dir)
    train_list = list()
    test_list = list()
    for cam_dir in cam_dir_list:
        if cam_dir != "../data/Cam253":
            continue
        video_list = file_io.get_listfile(cam_dir, ".avi")
        
        data_list = list()
        for file_name in video_list:
            file_dir_name = file_name.replace(".avi", "/")
            data_list += file_io.get_listfile(file_dir_name, data_ext)
        partition = 0.7
        train_data_len = int(len(data_list) * partition)

        random.shuffle(data_list)
        train_data = data_list[:train_data_len]
        test_data = data_list[train_data_len:]

        train_list += [d + " " + d.replace(data_ext, label_ext) for d in train_data]
        test_list += [d + " " + d.replace(data_ext, label_ext) for d in test_data]

    train_file_list_name = 'train_list5.txt'
    file_io.save_file(train_list, file_list_dir + train_file_list_name, True)

    test_file_list_name = 'test_list5.txt'
    file_io.save_file(test_list, file_list_dir + test_file_list_name, True)


