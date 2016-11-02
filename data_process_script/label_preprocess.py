import file_io
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import xmltodict

dsize = (227, 227)

def get_point(one_line):
    one_line = one_line[1:-2]
    p = one_line.split(",")
    p = np.array([float(p_s) for p_s in p], np.float32)
    return p


def load_mask(mask_file_name):
    p_l = list() 
    with open(mask_file_name, "r") as f:
        text = f.read()
        l = text.split("\n")
        for i in xrange(4):
            p_l.append(get_point(l[i + 1]))
    p_l = np.array(p_l)
    return p_l


    #mask_pt = np.loadtxt(mask_file_name)
    #print(mask_pt)

def get_density_map(annot_name, mask_coord):
    assert(annot_name.endswith(".xml"))
    with open(annot_name) as xml_d:
    	doc = xmltodict.parse(xml_d.read())

    img = np.zeros(dsize, np.float32)
    
    def add_to_image(image, bbox, coord):
    	xmax = int(bbox['xmax'])
    	xmin = int(bbox['xmin'])
    	ymax = int(bbox['ymax'])
    	ymin = int(bbox['ymin'])

        xmax = int((xmax - coord[1]) * dsize[0]/ float(coord[0] - coord[1]))
        xmin = int((xmin - coord[1]) * dsize[0]/ float(coord[0] - coord[1]))
        ymax = int((ymax - coord[3]) * dsize[0]/ float(coord[2] - coord[3]))
        ymin = int((ymin - coord[3]) * dsize[0]/ float(coord[2] - coord[3]))
    	density = 1/ float((ymax - ymin + 1) * (xmax - xmin + 1))	
        image[ymin:ymax, xmin:xmax] += density
    	return image


    for vehicle in doc['annotation']['vehicle']:
        bbox = vehicle['bndbox'] 
        add_to_image(img, bbox, mask_coord)

    return img

def crop_image(image_name, mask_pts, save_data = False):
    image = cv2.imread(image_name)
    x_max = int(np.max(mask_pts[:,0]))
    x_min = int(np.min(mask_pts[:,0]))
    y_max = int(np.max(mask_pts[:,1]))
    y_min = int(np.min(mask_pts[:,1]))
    mask_coord = np.array([x_max, x_min, y_max, y_min])
    croped_image = image[y_min:y_max, x_min:x_max]
    resize_image = cv2.resize(croped_image, dsize)

    annot_name = image_name.replace(".jpg", ".xml")
    density_map = get_density_map(annot_name, mask_coord)
    if save_data:
        save_image_name = image_name.replace(".jpg", "") + "_resize227.jpg"
        cv2.imwrite(save_image_name, resize_image)
        density_map.tofile(save_image_name.replace(".jpg", ".desmap"))
    #cv2.imshow("resized", resize_image)
    #display_density_map = np.repeat(np.expand_dims(density_map,2) * 255, 3, axis = 2)
    #cv2.imshow("masked", display_density_map)
    #cv2.waitKey(0)




if __name__ == "__main__":
    mask_dir_list = file_io.get_dir_list("../data")
    for mask_dir in mask_dir_list:
        mask_list = file_io.get_listfile(mask_dir, ".msk")
        for mask in mask_list:
            image_dir_name = mask.replace(".msk", "")
            image_list = file_io.get_listfile(image_dir_name, "jpg")
            mask_pts = load_mask(mask)
            for img in image_list:
                if img.endswith("_resize.jpg") or img.endswith("_resize227.jpg"):
                        continue
                try:
                    crop_image(img, mask_pts, True)
                except:
                    print(img)
                    pass
