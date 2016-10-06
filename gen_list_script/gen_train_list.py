import os
import file_io

if __name__ == "__main__":
    file_list_dir = "../file_list/"
    data_ext = "mixed10"
    label_ext = "desmap"
    file_dir = "../data/Cam253/[Cam253]-2016_4_21_15h_150f/"

    data_list = file_io.get_listfile(file_dir, data_ext)
    file_list = [d + " " + d.replace(data_ext, label_ext) for d in data_list]

    file_list_name = 'train_list1.txt'
    file_io.save_file(file_list, file_list_dir + file_list_name, True)


