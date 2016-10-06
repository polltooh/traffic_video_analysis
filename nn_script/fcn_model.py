import tensorflow as tf
import model_func as mf

def inference(feature, output_shape, keep_prob, is_train):
    b, h, w, c= feature.get_shape().as_list()
    wd = 0.0
    deconv1_shape = [b, 116, 79, c]
    deconv1 = mf.deconvolution_2d_layer(feature, [7, 4, 128, c], [1,4,3,1], [b, 38, 25, 128], 'VALID', wd, 'deconv1')
    deconv2 = mf.deconvolution_2d_layer(deconv1, [5, 5, 64, 128], [1,3,3,1], [b, 116, 79, 64], 'VALID', wd, 'deconv2')
    deconv3 = mf.deconvolution_2d_layer(deconv2, [5, 5, 1, 64], [1,3,3,1], [b, 352, 240, 1], 'VALID', wd, 'deconv3')
    return deconv3

def test_infer_size(label):
    conv1 = mf.convolution_2d_layer(label, [5,5,1,1], [1,3,3,1], 'VALID', 0.0, 'conv1')
    print(conv1)
    conv2 = mf.convolution_2d_layer(conv1, [5,5,1,1], [1,3,3,1], 'VALID', 0.0, 'conv2')
    print(conv2)
    conv3 = mf.convolution_2d_layer(conv2, [7,4,1,1], [1,4,3,1], 'VALID', 0.0, 'conv3')
    print(conv3)
    

def loss(infer, label):
    l2_loss = mf.l2_loss(infer, label, 'SUM', 'l2_loss')
    tf.add_to_collection('losses', l2_loss)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    
def train_op(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 1.0)
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op
