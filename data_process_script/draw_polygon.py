import numpy
from PIL import Image, ImageDraw
import cv2

polygon = [(1,1),(100,20),(120, 130), (12, 120)]
width = 299
height = 299

img = Image.new('L', (width, height), 0)
ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
mask = numpy.array(img)
print(mask)
cv2.imshow("mask", mask * 255)
cv2.waitKey(0)

