import tensorflow as tf
from tensorflow.python.training import moving_averages


FLAGS = tf.app.flags.FLAGS

def _variable_on_cpu(name, shape, initializer, trainable = True):
    """Helper to create a Variable stored on CPU memory.
    
    Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
    
    Returns:
            Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, trainable = trainable)
    return var

def _variable_with_weight_decay(name, shape, wd = 0.0):
    """Helper to create an initialized Variable with weight decay.
    
    Note that the Variable is initialized with a xavier initialization.
    A weight decay is added only if one is specified.
    
    #Args:
            name: name of the variable
            shape: list of ints
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                    decay is not added for this Variable.
   
    Returns:
            Variable Tensor
    """
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    # print("change var")
    # var = tf.Variable(tf.truncated_normal(shape, mean= 0.0, stddev = 1.0), name = name)
    if wd != 0.0:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _conv2d(x, w, b, strides, padding):
    return tf.nn.bias_add(tf.nn.conv2d(x, w,strides=strides, padding = padding), b)

def _conv3d(x, w, b, strides = [1,1,1,1,1], padding = 'SAME'):
    return tf.nn.bias_add(tf.nn.conv3d(x, w,strides=strides, padding = padding), b)

def add_leaky_relu(hl_tensor, leaky_param):
    return tf.maximum(hl_tensor, tf.mul(leaky_param, hl_tensor))

def _deconv2d(x, w, b, output_shape, strides, padding):
    return tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, output_shape, strides, padding), b)


def _add_leaky_relu(hl_tensor, leaky_param):
    """ add leaky relu layer
        Args:
            leaky_params should be from 0.01 to 0.1
    """
    return tf.maximum(hl_tensor, tf.mul(leaky_param, hl_tensor))

def _max_pool(x, ksize, strides):
    """ 2d pool layer"""
    pool = tf.nn.max_pool(x, ksize=ksize, strides= strides,
            padding='VALID')
    return pool

def _max_pool3(x, ksize, strides, name):
    """ 3d pool layer"""
    pool = tf.nn.max_pool3d(x, ksize=ksize, strides= strides,
        padding='VALID', name = name)
    return pool

def _avg_pool3(x, ksize, strides, name):
    """ 3d average pool layer """
    pool = tf.nn.avg_pool3d(x, ksize = ksize, strides = strides,
            padding = 'VALID', name = name)
    return pool

def _batch_norm(inputs, decay = 0.999, center = True, scale = False, epsilon = 0.001, 
				moving_vars = 'moving_vars', activation = None, is_training = None, 
				trainable = True, restore = True, scope = None, reuse = None):
    """ Copied from slim/ops.py 
        Adds a Batch Normalization layer. 
        Args:
    
            inputs: a tensor of size [batch_size, height, width, channels]
                    or [batch_size, channels].
            decay: decay for the moving average.
            center: If True, subtract beta. If False, beta is not created and ignored.
            scale: If True, multiply by gamma. If False, gamma is
                    not used. When the next layer is linear (also e.g. ReLU), this can be
                    disabled since the scaling can be done by the next layer.
            epsilon: small float added to variance to avoid dividing by zero.
            moving_vars: collection to store the moving_mean and moving_variance.
            activation: activation function.
            is_training: a placeholder whether or not the model is in training mode.
            trainable: whether or not the variables should be trainable or not.
            restore: whether or not the variables should be marked for restore.
            scope: Optional scope for variable_op_scope.
            reuse: whether or not the layer and its variables should be reused. To be
                            able to reuse the layer scope must be given.

        Returns:
            a tensor representing the output of the operation.
    """
    inputs_shape = inputs.get_shape()
    with tf.variable_op_scope([inputs], scope, 'BatchNorm', reuse = reuse):
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]
        beta, gamma = None, None

        if center:
                beta = _variable_on_cpu('beta', params_shape, tf.zeros_initializer)
        if scale:
                gamma = _variable_on_cpu('gamma', params_shape, tf.ones_initializer)

        # moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
        moving_mean = _variable_on_cpu('moving_mean', params_shape,tf.zeros_initializer, trainable = False)
        # tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, moving_mean)
        moving_variance = _variable_on_cpu('moving_variance', params_shape, tf.ones_initializer, trainable = False)
        # tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, moving_variance)
        
        def train_phase():
            mean, variance = tf.nn.moments(inputs, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, 
                                                            variance, decay)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)

        def test_phase():
            return moving_mean, moving_variance	

        mean, variance = tf.cond(is_training, train_phase, test_phase)
        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape()) 

        if activation:
            outputs = activation(outputs)

        return outputs

def batch_norm(x, is_train, layer_name):
    with tf.variable_scope(layer_name):
        bn = _batch_norm(x, is_training = is_train)
    return bn

def triplet_loss(infer, labels, radius = 2.0):
    """
    Args:
        infer: inference concatenate together with 2 * batch_size
        labels: 0 or 1 with batch_size
        radius:
    Return:
        loss: triplet loss
    """
            
    feature_1, feature_2 = tf.split(0,2,infer)

    feature_diff = tf.reduce_sum(tf.square(feature_1 - feature_2), 1)
    feature_list = tf.dynamic_partition(feature_diff, labels, 2)

    pos_list = feature_list[1]
    neg_list  = (tf.maximum(0.0, radius * radius - feature_list[0]))
    full_list = tf.concat(0,[pos_list, neg_list])
    loss = tf.reduce_mean(full_list)

    return loss

def l1_reg(input_tensor, weights):
    l1_reg_loss = tf.mul(tf.reduce_sum(tf.abs(input_tensor)), weights, name = "l1_reg_loss")
    tf.add_to_collection('losses', l1_reg_loss)

def l2_loss(infer, label, loss_type, layer_name):
    """
    Args:
        loss_type: 'SUM', 'MEAN'
            'SUM' uses reduce_sum
            'MEAN' uses reduce_mean
    """
    assert(loss_type == 'SUM' or loss_type == 'MEAN')
    with tf.variable_scope(layer_name):
        if loss_type == 'SUM':
            loss = tf.reduce_sum(tf.square(infer - label))
        else:
            loss = tf.reduce_mean(tf.square(infer - label))

    return loss


def huber_loss(infer, label, epsilon, layer_name):
    """
    Args:
        infer
        label
        epsilon
        layer_name
    """
    with tf.variable_scope(layer_name):
        abs_diff = tf.abs(tf.sub(infer, label));
        index = tf.to_int32(abs_diff <= epsilon, name = 'partition_index')
        l1_part, l2_part = tf.dynamic_partition(abs_diff, index, 2)
        #l1_loss = tf.reduce_mean(l1_part, name = 'l1_loss')
        #l2_loss = tf.reduce_mean(tf.square(l2_part), name = 'l2_loss')
        l1_part_loss = epsilon * (l1_part - 0.5 * epsilon)
        l2_part_loss = 0.5 * tf.square(l2_part)
        hloss = tf.reduce_mean(tf.concat(0, [l1_part_loss,l2_part_loss]), 
                    name = 'huber_loss_sum')
    return hloss

def convolution_2d_layer(x, kernel_shape, kernel_stride, padding, wd, layer_name):
    """
    Args:
        x
        kernel_shape: [height, width, input_channel, output_channel]
        kernel_stride: [height, width]
        padding: "SAME" or "VALID"
        wd: weight decay params
        layer_name: 
    """
    with tf.variable_scope(layer_name):
        weights = _variable_with_weight_decay('weights', kernel_shape, wd)
        biases = _variable_on_cpu('biases', [kernel_shape[-1]], tf.constant_initializer(0.0))
        conv = _conv2d(x, weights, biases, [1, kernel_stride[0], kernel_stride[1],1], padding)

    return conv

def fully_connected_layer(x, output_num, wd, layer_name):
    """
    Args:
        x
        output_num 
        wd
        layer_num
    """
    input_shape = x.get_shape().as_list()
    if len(input_shape) > 2:
        x = tf.reshape(x, [input_shape[0], -1])

    input_shape = x.get_shape().as_list()

    with tf.variable_scope(layer_name):
        weights = _variable_with_weight_decay("weights", [input_shape[1], output_num], wd)
        biases = _variable_on_cpu('biases', [output_num], tf.constant_initializer(0.0))
        fc = tf.matmul(x, weights) + biases
    return fc

def deconvolution_2d_layer(x, kernel_shape, kernel_stride, output_shape, padding, wd, layer_name):
    """
    Args:
        x
        kernel_shape: [height, width, output_channel, input_channel]
        kernel_stride: [height, width]
        output_shape: [batch_size, height, width, channel]
        padding: "SAME" or "VALID"
        wd: weight decay params
        layer_name: 
    """
    with tf.variable_scope(layer_name):
        weights = _variable_with_weight_decay('weights', kernel_shape, wd)
        biases = _variable_on_cpu('biases', [kernel_shape[-2]], tf.constant_initializer(0.0))
        deconv = _deconv2d(x, weights, biases, output_shape, [1, kernel_stride[0], kernel_stride[1], 1], padding)
    return deconv

def maxpool_2d_layer(x, kernel_shape, kernel_stride, layer_name):
    """
    Args:
        x
        kernel_shape: [height, weights]
        kernel_stride: [height, weights]
    """

    with tf.variable_scope(layer_name):
        max_pool = _max_pool(x, [1, kernel_shape[0], kernel_shape[1], 1], [1, kernel_stride[0], kernel_stride[1], 1])
    return max_pool


def res_layer(x, kernel_shape, kernel_stride, padding, wd, layer_name, repeat_num, leaky_param = 0.01, is_train = None):
    """
    Args:
        x
        kernel_shape: [height, weights, input_channel, ouput_channel]
        kernel_stride: [height, weights]
        padding: SAME or VALID
        is_train: a tensor indicate is train or not
    """
    with tf.variable_scope(layer_name):
        conv = tf.identity(x)
        for i in xrange(repeat_num):
            conv = convolution_2d_layer(conv, kernel_shape, 
                        kernel_stride, padding, wd, "_%d"%i)
            if is_train is not None:
                conv = _batch_norm(conv, is_training = is_train)
            conv = add_leaky_relu(conv, leaky_param)

        final_conv = tf.add(conv,x, 'res_connect')
    return final_conv

def res_pad(x, input_channel, output_channel, layer_name):
    """
    Args:
        x
        input_channel: a number
        output_channel: a number
        layer_name
    """
    with tf.variable_scope(layer_name):
        forward_pad = (output_channel - input_channel) // 2
        backward_pad = output_channel - input_channel - forward_pad
        x_pad = tf.pad(x, [[0,0],[0,0],[0,0],[forward_pad, backward_pad]])
    return x_pad


def copy_layer(x, layer_handle, repeat_num, layer_name, *params):
    """
    Args:
        x
        layer_handle: function handler
        repeat_num: the number of repeat
        layer_name
        params: parameters for the function
    """
    for i in xrange(repeat_num):
        with tf.variable_scope(layer_name + "_%d"%i):
            x = layer_handle(x, *params)
    return x

def unpooling_layer(x, output_size, layer_name):
    """ Bilinear Interpotation resize 
    Args:
        x
        output_size [image_height, image_width]
        layer_name 
    """
    with tf.variable_scope(layer_name):
        return tf.image.resize_images(x, output_size[0], output_size[1])

def atrous_convolution_layer(x, kernel_shape, rate, padding, wd, layer_name):
    """
    Args:
            x
            kernel_shape
            rate
            padding
            wd
            layer_name
    """
    with tf.variable_scope(layer_name):
        weights = _variable_with_weight_decay('weights', kernel_shape, wd)
        biases = _variable_on_cpu('biases', [kernel_shape[-1]], tf.constant_initializer(0.0))
        atrous_conv = tf.nn.atrous_conv2d(x, weights, rate, padding, name=None)
	return atrous_conv

