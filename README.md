# traffic_video_analysis

This work is the official codes for paper: S. Zhang, G. Wu, J. Costeira, J. Moura, “Deep Understanding of Traffic Density from Large-Scale Web Camera Data”, accepted by Conference on Computer Vision and Pattern Recognition (CVPR) 2017.

# Please email the author citycam12@gmail.com if you are interested in the training code.

# Please cite the following paper if you found the code is helpful for you:
S. Zhang, G. Wu, J. Costeira, J. Moura, “Understanding Traffic Density from Large-Scale Web Camera Data”, accepted by Conference on Computer Vision and Pattern Recognition (CVPR) 2017.

@article{szhangwebCam,
  title={Understanding Traffic Density from Large-Scale Web Camera Data},
  author={Gon{S. Zhang, G. Wu, J. Costeira, J. Moura},
  journal={arXiv preprint arXiv:1703.05868},
  year={2017}
}


# Usage:
### caffe_script:
extract_hypercolumn.py: The caffe script for extracting hypercolumn features. 
It uses the resnet-152 and combine res2a, res3a and res4a feature into a feature
volumn called hypercolumn.
Please set the root of caffe inside the script. It will read the images stored 
in the ../file_list/image_name_list.txt. Please feel free to change it to your 
file location. It assume the image format is jpg and save the hypercolumn feature
with the same name but with extention ".resnet_hypercolumn"
