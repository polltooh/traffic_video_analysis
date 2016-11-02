import file_io
import os

if __name__ == "__main__":
    avi_dir_list = file_io.get_dir_list("../data")
    for avi_dir in avi_dir_list:
        avi_file_list = file_io.get_listfile(avi_dir, ".avi")
        avi_file_list.sort()
        for avi in avi_file_list:
            image_dir = avi.replace(".avi", "")
            command = "ffmpeg -i " + avi + " " + image_dir + "/%06d.jpg"
            os.system(command)
