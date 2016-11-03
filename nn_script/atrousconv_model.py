import tensorflow as tf
import model_func as mf

def inference(image, output_shape, keep_prob, is_train):
    wd = 0.0004
    leaky_param = 0.01
    
    aconv1 = mf.atrous_convolution_layer(image, [3,3,3,64], 2, 'SAME', wd, 'aconv1')
    aconv1_relu = mf.add_leaky_relu(aconv1, leaky_param)

    aconv2 = mf.atrous_convolution_layer(aconv1_relu, [3,3,64,128], 2, 'SAME', wd, 'aconv2')
    aconv2_relu = mf.add_leaky_relu(aconv2, leaky_param)

    aconv3 = mf.atrous_convolution_layer(aconv2_relu, [3,3,128,256], 2, 'SAME', wd, 'aconv3')
    aconv3_relu = mf.add_leaky_relu(aconv3, leaky_param)

    aconv4 = mf.atrous_convolution_layer(aconv3_relu, [3,3,256,128], 2, 'SAME', wd, 'aconv4')
    aconv4_relu = mf.add_leaky_relu(aconv4, leaky_param)

    aconv5 = mf.atrous_convolution_layer(aconv4_relu, [3,3,128,64], 2, 'SAME', wd, 'aconv5')
    aconv5_relu = mf.add_leaky_relu(aconv5, leaky_param)

    aconv6 = mf.atrous_convolution_layer(aconv5_relu, [3,3,64,1], 2, 'SAME', wd, 'aconv6')
    aconv6_relu = mf.add_leaky_relu(aconv6, leaky_param = 0.0)

    fc1 = mf.fully_connected_layer(aconv6_relu, 1000, wd, "fc1")
    fc1_relu = mf.add_leaky_relu(fc1, leaky_param)
    fc2 = mf.fully_connected_layer(fc1_relu, 1, wd, "fc2")
    fc2_relu = mf.add_leaky_relu(fc2, leaky_param)

    return aconv6_relu, fc2_relu

def loss(infer, count_diff_infer, label):
    l2_loss = tf.reduce_mean(tf.reduce_sum(tf.square(infer - label), [1,2,3]), name = 'l2_loss')
    count_infer = tf.add(count_diff_infer, tf.reduce_sum(infer, [1,2,3]), name = "count_infer")
    c_lambda = 0.01
    count_loss = tf.mul(c_lambda, tf.reduce_mean(tf.square(count_infer - tf.reduce_sum(label, [1,2,3]))),
                    name = 'count_loss')

    tf.add_to_collection('losses', count_loss)
    tf.add_to_collection('losses', l2_loss)

    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    
def train_op(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 1.0)
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op
