import numpy as np
import cv2

def load_image(image_name):
    image = cv2.imread(image_name)
    return image

def norm_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image

def repeat_image(image):
    #image = np.expand_dims(image, 2)
    image = np.tile(image, (1,1,3))
    return image

def show_image(input_image, normalize = True, wait_time = 0, image_name = "image"):
    if normalize:
        input_image = norm_image(input_image)
    cv2.imshow(image_name, input_image)
    cv2.waitKey(wait_time)

def save_image(input_image, image_name, is_norm = False):
    if is_norm:
        input_image = norm_image(input_image)
    cv2.imwrite(image_name, input_image)
    #cv2.imwrite(FLAGS.image_dir + "/%08d.jpg"%(i/100), image)

def resize_image(image, rshape):
    """
    Args:
        image:
        rshape: (new_image_hegiht, new_image_width)
    """
    return cv2.resize(image, rshape)
