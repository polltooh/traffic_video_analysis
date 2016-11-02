import file_io



if __name__ == "__main__":
    file_list_dir = "../file_list/"
    data_ext = "_resize.jpg"
    label_ext = "_resize.desmap"
    file_dir = "../data/"
    
    cam_dir_list = file_io.get_dir_list(file_dir)
    data_list = list()
    for cam_dir in cam_dir_list:
        video_list = file_io.get_listfile(cam_dir, ".avi")
        
        for file_name in video_list:
            file_dir_name = file_name.replace(".avi", "/")
            data_list += file_io.get_listfile(file_dir_name, data_ext)

    file_io.save_file(data_list, file_list_dir + 'image_name_list.txt', False)
