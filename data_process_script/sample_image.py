import file_io
import os

if __name__ == "__main__":
    avi_dir = "/home/mscvadmin/traffic_video_analysis/data/Cam253/"
    avi_file_list = file_io.get_listfile(avi_dir, ".avi")
    avi_file_list.sort()
    for avi in avi_file_list:
        if avi == avi_dir + "[Cam253]-2016_4_21_15h_150f.avi":
            continue
        image_dir = avi.replace(".avi", "")
        command = "ffmpeg -i " + avi + " " + image_dir + "/%06d.jpg"
        os.system(command)
