import tensorflow as tf
import model_func as mf

def inference(feature, output_shape, keep_prob, is_train):
    b, h, w, c= feature.get_shape().as_list()
    wd = 0.0
    deconv1_shape = [b, 116, 79, c]
    deconv1 = mf.deconvolution_2d_layer(feature, [3, 3, 128, c], [1,2,2,1], [b, 17, 17, 128], 'VALID', wd, 'deconv1')
    deconv2 = mf.deconvolution_2d_layer(deconv1, [3, 3, 128, 128], [1,2,2,1], [b, 36, 36, 128], 'VALID', wd, 'deconv2')
    deconv3 = mf.deconvolution_2d_layer(deconv2, [3, 3, 128, 128], [1,2,2,1], [b, 74, 74, 128], 'VALID', wd, 'deconv3')
    deconv4 = mf.deconvolution_2d_layer(deconv3, [3, 3, 128, 128], [1,2,2,1], [b, 149, 149, 128], 'VALID', wd, 'deconv4')
    deconv5 = mf.deconvolution_2d_layer(deconv4, [3, 3, 1, 128], [1,2,2,1], [b, 299, 299, 1], 'VALID', wd, 'deconv5')
    return deconv5

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
