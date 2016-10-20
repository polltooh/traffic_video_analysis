import tensorflow as tf
import model_func as mf

def inference(feature, output_shape, keep_prob, is_train):
    b, h, w, c= feature.get_shape().as_list()
    wd = 0.0
    leaky_param = 0.01
    deconv1_shape = [b, 116, 79, c]
    deconv1 = mf.deconvolution_2d_layer(feature, [3, 3, 512, c], [2,2], [b, 17, 17, 512], 'VALID', wd, 'deconv1')
    deconv1_relu = mf.add_leaky_relu(deconv1, leaky_param)

    deconv2 = mf.deconvolution_2d_layer(deconv1_relu, [3, 3, 256, 512], [2,2], [b, 36, 36, 256], 'VALID', wd, 'deconv2')
    deconv2_relu = mf.add_leaky_relu(deconv2, leaky_param)

    deconv3 = mf.deconvolution_2d_layer(deconv2_relu, [3, 3, 128, 256], [2,2], [b, 74, 74, 128], 'VALID', wd, 'deconv3')
    deconv3_relu = mf.add_leaky_relu(deconv3, leaky_param)

    deconv4 = mf.deconvolution_2d_layer(deconv3_relu, [3, 3, 64, 128], [2,2], [b, 149, 149, 64], 'VALID', wd, 'deconv4')
    deconv4_relu = mf.add_leaky_relu(deconv4, leaky_param)

    deconv5 = mf.deconvolution_2d_layer(deconv4_relu, [3, 3, 32, 64], [2,2], [b, 299, 299, 32], 'VALID', wd, 'deconv5')
    deconv5_relu = mf.add_leaky_relu(deconv5, leaky_param)

    deconv6 = mf.deconvolution_2d_layer(deconv5, [3, 3, 1, 32], [1,1], [b, 299, 299, 1], 'SAME', wd, 'deconv6')

    return deconv6

def test_infer_size(label):
    conv1 = mf.convolution_2d_layer(label, [3,3,1,1], [1,2,2,1], 'VALID', 0.0, 'conv1')
    print(conv1)
    conv2 = mf.convolution_2d_layer(conv1, [3,3,1,1], [1,2,2,1], 'VALID', 0.0, 'conv2')
    print(conv2)
    conv3 = mf.convolution_2d_layer(conv2, [3,3,1,1], [1,2,2,1], 'VALID', 0.0, 'conv3')
    print(conv3)
    conv4 = mf.convolution_2d_layer(conv3, [3,3,1,1], [1,2,2,1], 'VALID', 0.0, 'conv4')
    print(conv4)
    conv5 = mf.convolution_2d_layer(conv4, [3,3,1,1], [1,2,2,1], 'VALID', 0.0, 'conv5')
    print(conv5)
    exit(1) 

def loss(infer, label):
    l2_loss = mf.l2_loss(infer, label, 'SUM', 'l2_loss')
    tf.add_to_collection('losses', l2_loss)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    
def train_op(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 1.0)
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op
