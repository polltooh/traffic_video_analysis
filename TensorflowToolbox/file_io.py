import os 
import random
import numpy as np

def get_listfile(image_dir, extension = ".jpg"):
    if not image_dir.endswith("/"):
        image_dir = image_dir + "/"

    image_list = os.listdir(image_dir)
    image_list = [image_dir + image for image in image_list if image.endswith(extension)]
    return image_list

def get_dir_list(frame_dir):
    if not frame_dir.endswith("/"):
        frame_dir = frame_dir + "/"

    dir_list = os.listdir(frame_dir)
    dir_list = [frame_dir + image_dir for image_dir in dir_list]
    return dir_list

def delete_last_empty_line(s):
    end_index = len(s) - 1
    while(end_index >= 0 and (s[end_index] == "\n" or s[end_index] == "\r")):
        end_index -= 1
    s = s[:end_index + 1]	
    return s

def read_file(file_name):
    with open(file_name, "r") as f:
        s = f.read();
        s = delete_last_empty_line(s)
        s_l = s.split("\n")
    return s_l

def save_file(string_list, file_name, shuffle_data = False):
    if (shuffle_data):
        random.shuffle(string_list)

    with open(file_name, "w") as f:
        file_string = "\n".join(string_list)
        if (file_string[-1] != "\n"):
                file_string += "\n"
        f.write(file_string)

def get_file_length(file_name):
    with open(file_name, 'r') as f:
        s = f.read()
        s_l = s.split("\n")
        total_len = len(s_l)
    return total_len

def save_numpy_array(numpy_array, file_name):
    numpy_array.tofile(file_name)

def remove_extension(file_name):
    index = file_name.rfind(".")
    if (index == -1):
        return file_name
    else:
        return file_name[0:index]
