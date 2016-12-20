import numpy as np
import cv2
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_conf_mat(densmap_name):
    fig = plt.figure(figsize = (20,20))
    plt.clf()
    ax = fig.add_subplot(111)
    #ax.set_aspect(1)
    densmap = np.fromfile(densmap_name, np.float32)
    densmap = densmap.reshape(227, 227)
    densmap *= 100
    densmap[densmap > 1] = 1
    res = ax.imshow(densmap, cmap = plt.cm.jet,
            interpolation = 'nearest')

    plt.savefig('density.jpg')
    img = cv2.imread("density.jpg")
    img = cv2.resize(img, (227,227))
    cv2.imshow("i", img)#
    cv2.waitKey(0)
    #plt.show()

def plot_3d(densmap_name):
    densmap = np.fromfile(densmap_name, np.float32) * 100
    #densmap = densmap.reshape(227, 227)
    
    x = np.arange(0, 227, 1)
    y = np.arange(0, 227, 1)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(xx, yy, densmap, cmap=plt.cm.jet, linewidth=0.2)
    plt.show()

def norm_image(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def opencv_plot(des_name):
    densmap = np.fromfile(densmap_name, np.float32)
    densmap = np.reshape(densmap, (227, 227))
    #densmap = norm_image(densmap) * 100 
    densmap *= 100.0
    densmap[densmap >1 ] = 1
    densmap = norm_image(densmap) * 255
    densmap = densmap.astype(np.uint8)
    im_color = cv2.applyColorMap(densmap, cv2.COLORMAP_JET)
    cv2.imshow("im", im_color)
    cv2.waitKey(0)


densmap_name = "/home/mscvadmin/traffic_video_analysis/nn_script/resdeconv_results/[Cam253]-2016_5_3_18h_200f_000102_resize.npy"

opencv_plot(densmap_name)
#plot_conf_mat(densmap_name)
#@plot_3d(densmap_name)
