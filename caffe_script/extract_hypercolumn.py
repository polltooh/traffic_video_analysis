import numpy as np
import os
import sys
import caffe
import tensorflow as tf
import file_io
import cv2

# Please set the root of caffe
caffe_root = ""

if caffe_root == "":
    print("Please set the root of caffe")
    exit(1)

file_list_name = "../file_list/image_name_list.txt"

if os.path.isfile(caffe_root + 'models/resnet/ResNet-152-model.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'

batch_size = 1
caffe.set_mode_cpu()

model_def = caffe_root + 'models/resnet/ResNet-152-deploy.prototxt'
model_weights = caffe_root + 'models/resnet/ResNet-152-model.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension

transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,3, 224, 224)

output_shape = [56, 56]

def resize_image(np_array, image_size):
    input_shape = np_array.shape
    new_shape = image_size + [input_shape[2]]
    new_array = np.zeros((new_shape), np.float32)
    for i in xrange(input_shape[2]):
        new_array[:,:,i] = cv2.resize(np_array[:,:,i], tuple(image_size))
    return new_array

def process_image(image_name):
    res2a = net.blobs['res2a'].data.squeeze().transpose((1,2,0))
    res3a = net.blobs['res3a'].data.squeeze().transpose((1,2,0))
    res4a = net.blobs['res4a'].data.squeeze().transpose((1,2,0))

    image = caffe.io.load_image(image_name)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    net.forward()
    res2a = resize_image(res2a, output_shape)
    res3a = resize_image(res3a, output_shape)
    res4a = resize_image(res4a, output_shape)
    """ shape of hypercolumn is (56, 56, 1792) """
    hypercolumn = np.concatenate((res2a, res3a, res4a), 2)
    return hypercolumn

image_name_list = file_io.read_file(file_list_name)
count = 0
for image_name in image_name_list:
    hypercolumn = process_image(image_name)
    feature_name = image_name.replace(".jpg", ".resnet_hypercolumn")
    hypercolumn.tofile(feature_name)
    count = count + 1
    print("count: %d / %d"%(count , len(image_name_list)))
