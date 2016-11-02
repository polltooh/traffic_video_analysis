import tensorflow as tf
import model_func as mf

def inference(image, output_shape, keep_prob, is_train):
    wd = 0.0001
    leaky_param = 0.01
    input_shape = image.get_shape().as_list()
    conv1 = mf.convolution_2d_layer(image, [5, 5,3,64], [1,1], 'SAME', wd, 'conv1') 
    conv1_pool = mf.maxpool_2d_layer(conv1, [2,2], [2,2], 'maxpool1')

    res1 = mf.copy_layer(conv1_pool, mf.res_layer, 3, 'res', [3,3, 64, 64], [1,1], "SAME", wd, 'res_1', 2, leaky_param, is_train)
    res1_pool = mf.maxpool_2d_layer(res1, [2,2], [2,2], 'maxpool2')
    res1_pad = mf.res_pad(res1_pool, 64, 128, 'res_pad1')
    
    res2 = mf.copy_layer(res1_pad, mf.res_layer, 4, 'res', [3,3, 128, 128], [1,1], "SAME", wd, 'res_2', 2, leaky_param, is_train)
    res2_pool = mf.maxpool_2d_layer(res2, [2,2], [2,2], 'maxpool2')
    res2_pad = mf.res_pad(res2_pool, 128, 256, 'res_pad2')

    res3 = mf.copy_layer(res2_pad, mf.res_layer, 6, 'res', [3,3, 256, 256], [1,1], "SAME", wd, 'res_3', 2, leaky_param, is_train)
    #res3_pool = mf.maxpool_2d_layer(res3, [2,2], [2,2], 'maxpool3')
    #res3_pad = mf.res_pad(res3_pool, 256, 512, 'res_pad3')

    #res4 = mf.copy_layer(res3_pad, mf.res_layer, 3, 'res', [3,3, 512, 512], [1,1], "SAME", wd, 'res_4', 2, leaky_param, is_train)

    res1_unpool = mf.unpooling_layer(res1, [input_shape[1], input_shape[2]], 'res1_unpool')
    res2_unpool = mf.unpooling_layer(res2, [input_shape[1], input_shape[2]], 'res2_unpool')
    res3_unpool = mf.unpooling_layer(res3, [input_shape[1], input_shape[2]], 'res3_unpool')
    #res4_unpool = mf.unpooling_layer(res4, [input_shape[1], input_shape[2]], 'res4_unpool')
    hypercolumn = tf.concat(3, [res1_unpool, res2_unpool, res3_unpool], name = 'hypercolumn')

    output_shape = input_shape
    output_shape[3] = 1
    #heatmap = mf.deconvolution_2d_layer(hypercolumn, [1,1,1,hypercolumn.get_shape()[3]], [1,1], output_shape, 'SAME', wd, 'heatmap')
    heatmap = mf.convolution_2d_layer(hypercolumn, [1,1,hypercolumn.get_shape()[3],1], [1,1], 'SAME', wd, 'heatmap')
    return heatmap


def loss(infer, label):
    l2_loss = mf.l2_loss(infer, label, 'SUM', 'l2_loss')
    tf.add_to_collection('losses', l2_loss)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')
    
def train_op(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 1.0)
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op
