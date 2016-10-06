import tensorflow as tf
import model_func as mf

def inference(feature, output_shape, keep_prob, is_train):
    feature_shape = feature.get_shape().as_list()
    wd = 0.0
    deconv1 = mf.deconvolution_2d_layer(feature, [3, 3, 1, feature_shape[3]], [1,1,1,1], output_shape, wd, 'deconv1')
    return deconv1

def loss(infer, label):
    l2_loss = mf.l2_loss(infer, label, 'l2_loss')
    tf.add_to_collection('losses', l2_loss)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    
def train_op(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 1.0)
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op
