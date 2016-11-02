import tensorflow as tf
import model_func as mf

def inference(feature, output_shape, keep_prob, is_train):
    b, h, w, c= feature.get_shape().as_list()
    wd = 0.0004
    leaky_param = 0.01

    deconv11 = mf.deconvolution_2d_layer(feature, [3, 3, 256, c], [1,1], [b, 56, 56, 256], 'SAME', wd, 'deconv11')
    deconv11_relu = mf.add_leaky_relu(deconv11, leaky_param)

    deconv12 = mf.deconvolution_2d_layer(deconv11_relu, [3, 3, 256, 256], [1,1], [b, 56, 56, 256], 'SAME', wd, 'deconv12')
    deconv12_relu = mf.add_leaky_relu(deconv12, leaky_param)

    deconv21 = mf.deconvolution_2d_layer(deconv12_relu, [3, 3, 128, 256], [2,2], [b, 113, 113, 128], 'VALID', wd, 'deconv21')
    deconv21_relu = mf.add_leaky_relu(deconv21, leaky_param)
    deconv22 = mf.deconvolution_2d_layer(deconv21_relu, [3, 3, 128, 128], [1,1], [b, 113, 113, 128], 'SAME', wd, 'deconv22')
    deconv22_relu = mf.add_leaky_relu(deconv22, leaky_param)
    
    deconv31 = mf.deconvolution_2d_layer(deconv22_relu, [3, 3, 64, 128], [2,2], [b, 227, 227, 64], 'VALID', wd, 'deconv31')
    deconv31_relu = mf.add_leaky_relu(deconv31, leaky_param = 0.0)
    deconv32 = mf.deconvolution_2d_layer(deconv31_relu, [3, 3, 1, 64], [1,1], [b, 227, 227, 1], 'SAME', wd, 'deconv32')
    #conv1x1 = mf.convolution_2d_layer(deconv31_relu, [1,1,64,1], [1,1],"VALID", wd, 'conv1x1')
   

    #feature_pad = tf.pad(feature, [[0,0],[9,9],[9,9],[0, 0]])
    #deconv11 = mf.deconvolution_2d_layer(feature_pad, [3, 3, 128, c], [1,1], [b, 74, 74, 128], 'SAME', wd, 'deconv11')
    #deconv11_relu = mf.add_leaky_relu(deconv11, leaky_param)

    #deconv12 = mf.deconvolution_2d_layer(deconv11_relu, [3, 3, 128, 128], [1,1], [b, 74, 74, 128], 'SAME', wd, 'deconv12')
    #deconv12_relu = mf.add_leaky_relu(deconv12, leaky_param)

    #deconv21 = mf.deconvolution_2d_layer(deconv12_relu, [3, 3, 64, 128], [2,2], [b, 149, 149, 64], 'VALID', wd, 'deconv21')
    #deconv21_relu = mf.add_leaky_relu(deconv21, leaky_param)
    #deconv22 = mf.deconvolution_2d_layer(deconv21_relu, [3, 3, 64, 64], [1,1], [b, 149, 149, 64], 'SAME', wd, 'deconv22')
    #deconv22_relu = mf.add_leaky_relu(deconv22, leaky_param)
    #
    #deconv31 = mf.deconvolution_2d_layer(deconv22_relu, [3, 3, 32, 64], [2,2], [b, 299, 299, 32], 'VALID', wd, 'deconv31')
    #deconv31_relu = mf.add_leaky_relu(deconv31, leaky_param)
    #deconv32 = mf.deconvolution_2d_layer(deconv31_relu, [3, 3, 1, 32], [1,1], [b, 299, 299, 1], 'SAME', wd, 'deconv32')
    
    fc1 = mf.fully_connected_layer(deconv32, 1000, wd, "fc1")
    #fc1 = mf.fully_connected_layer(conv1x1, 1000, wd, "fc1")
    fc1_relu = mf.add_leaky_relu(fc1, leaky_param)
    fc2 = mf.fully_connected_layer(fc1_relu, 1, wd, "fc2")
    fc2_relu = mf.add_leaky_relu(fc2, leaky_param)

    return conv1x1, fc2_relu

def test_infer_size(label):
    conv1 = mf.convolution_2d_layer(label, [3,3,1,1], [2,2], 'VALID', 0.0, 'conv1')
    print(conv1)
    conv2 = mf.convolution_2d_layer(conv1, [3,3,1,1], [2,2], 'VALID', 0.0, 'conv2')
    print(conv2)
    #conv3 = mf.convolution_2d_layer(conv2, [3,3,1,1], [2,2], 'VALID', 0.0, 'conv3')
    #print(conv3)
    #conv4 = mf.convolution_2d_layer(conv3, [3,3,1,1], [2,2], 'VALID', 0.0, 'conv4')
    #print(conv4)
    #conv5 = mf.convolution_2d_layer(conv4, [3,3,1,1], [2,2], 'VALID', 0.0, 'conv5')
    #print(conv5)
    exit(1) 

def loss(infer, count_diff_infer, label):
    l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(infer - label), [1,2,3]), name = 'l2_loss')
    count_infer = tf.add(count_diff_infer, tf.reduce_sum(infer, [1,2,3]), name = "count_infer")
    c_lambda = 1.0
    count_loss = tf.mul(c_lambda, tf.reduce_mean(tf.square(count_infer - tf.reduce_sum(label, [1,2,3]))),
                    name = 'count_loss')

    tf.add_to_collection('losses', count_loss)
    tf.add_to_collection('losses', l2_loss)

    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    
def train_op(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 1.0)
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op
