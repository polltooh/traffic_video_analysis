import file_io
import os


def sample(avid_dir_list):
    avi_dir_list = file_io.get_dir_list(avid_dir_list)
    print(avi_dir_list)
    for avi_dir in avi_dir_list:
        avi_file_list = file_io.get_listfile(avi_dir, ".avi")
        avi_file_list.sort()
        for avi in avi_file_list:
            image_dir = avi.replace(".avi", "")
            command = "ffmpeg -i " + avi + " " + image_dir + "/%06d.jpg"
            os.system(command)


if __name__ == "__main__":
    #avid_dir_list = "../data/data_new/Training_Data"
    avid_dir_list = "/home/mscvadmin/Downloads/wd_new/Training_Data"
    #sample(avid_dir_list)
    #avid_dir_list = "../data/data_new/Testing_Data"
    avid_dir_list = "/home/mscvadmin/Downloads/wd_new/Testing_Data"
    sample(avid_dir_list)
