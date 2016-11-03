import scipy.io as sio
import numpy as np
import file_io
import cv2

mat_dir = "../data/toJeff/Train/Resized_GTdensity/"
mat_list = file_io.get_listfile(mat_dir, "mat")

for f in mat_list:
    mat = sio.loadmat(f)
    np_array = mat['gtDensities']
    np_array = np_array.astype(np.float32)
    np_array = cv2.resize(np_array, (227,227))
    np_name = f.replace(".mat",".npy")
    np_array.tofile(np_name)

