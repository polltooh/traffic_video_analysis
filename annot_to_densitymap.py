import xml.etree.ElementTree as ET
import os
import numpy as np
import xmltodict
import cv2


def show_mask(mask_image, car_image_name):
    car_image = cv2.imread(car_image_name)
    car_image = car_image.astype(np.float32)

    mask_rep = np.expand_dims(mask_image, axis = 2)
    mask_rep = np.repeat(mask_rep, 3, axis = 2)
    mask_rep *= 1/mask_rep.max()
    car_image = (car_image - car_image.min()) / (car_image.max() - car_image.min())
    car_image += mask_rep 
    car_image = (car_image - car_image.min()) / (car_image.max() - car_image.min())
    car_image = np.vstack((car_image,mask_rep))
    cv2.imshow("test", car_image)
    cv2.waitKey(0)

def get_density_map(data_name, save_data, show_image = False):
    assert(data_name.endswith(".xml"))
    #xml_data = data_name + '.xml'
    
    with open(data_name) as xml_d:
    	doc = xmltodict.parse(xml_d.read())

    img = np.zeros((240,352), np.float32)
    
    def add_to_image(image, bbox):
    	xmin = int(bbox['xmin'])
    	ymin = int(bbox['ymin'])
    	xmax = int(bbox['xmax'])
    	ymax = int(bbox['ymax'])
    	density = 1/ float((ymax - ymin) * (xmax - xmin))	
        image[ymin:ymax, xmin:xmax] += density
    	return image
    
    for vehicle in doc['annotation']['vehicle']:
    	add_to_image(img, vehicle['bndbox'])
    
    if show_image: 
        show_mask(img, data_name.replace('xml','jpg'))
    if save_data:
        img.tofile(data_name.replace('xml','desmap'))

if __name__ == "__main__":
    data_name = 'data/Cam253/[Cam253]-2016_4_21_15h_150f/000150.xml'
    path_name = 'data/Cam253/[Cam253]-2016_4_21_15h_150f/'
    list_name = os.listdir(path_name)
    for f in list_name:
        if f.endswith("xml"):
            data_name = path_name + f
            get_density_map(data_name, True, False)
