import xml.etree.ElementTree as ET
import numpy as np
import xmltodict
import matplotlib.pyplot as plt
#import cv2

xml_data = 'data/Cam253/[Cam253]-2016_4_21_15h_150f/000150.xml'

with open(xml_data) as xml_d:
	doc = xmltodict.parse(xml_d.read())

img = np.zeros((352, 240), np.float32)

def add_to_image(image, bbox):
	xmin = int(bbox['xmin'])
	ymin = int(bbox['ymin'])
	xmax = int(bbox['xmax'])
	ymax = int(bbox['ymax'])
	density = 1/ float((ymax - ymin) * (xmax - xmin))	

	image[xmin:xmax, ymin:ymax] += density
	print(np.sum(image))
	print(xmin)
	print(xmax)
	print(ymin)
	print(ymax)
	return image

for vehicle in doc['annotation']['vehicle']:
	add_to_image(img, vehicle['bndbox'])

imgplot = plt.imshow(img)
plt.show()
print(np.sum(img))
