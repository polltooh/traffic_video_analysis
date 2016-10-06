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
    
    Args:
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

def _conv2d(x, w, b, strides = [1,1,1,1]):
    return tf.nn.bias_add(tf.nn.conv2d(x, w,strides=strides, padding = 'SAME'), b)

def _conv3d(x, w, b, strides = [1,1,1,1,1], padding = 'SAME'):
    return tf.nn.bias_add(tf.nn.conv3d(x, w,strides=strides, padding = padding), b)

def add_leaky_relu(hl_tensor, leaky_param):
    return tf.maximum(hl_tensor, tf.mul(leaky_param, hl_tensor))

def _deconv2d(x, w, b, output_shape, strides = [1,1,1,1]):
    return tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, output_shape, strides), b)

def _unpooling(x, output_size):
    """ NEAREST_NEIGHBOR resize """
    return tf.image.resize_images(x, output_size[1], output_size[2],1)

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

def l2_loss(infer, label, layer_name):
    with tf.variable_scope(layer_name):
        loss = tf.reduce_mean(tf.square(infer - label))
    return loss

def convolution_2d_layer(x, kernel_shape, kernel_stride, wd, layer_name):
    with tf.variable_scope(layer_name):
        weights = _variable_with_weight_decay('weights', kernel_shape, wd)
        biases = _variable_on_cpu('biases', [kernel_shape[-1]], tf.constant_initializer(0.0))
        conv = _conv2d(x, weights, biases, strides =  kernel_stride)
    return conv

def deconvolution_2d_layer(x, kernel_shape, kernel_stride, output_shape, wd, layer_name):
    with tf.variable_scope(layer_name):
        weights = _variable_with_weight_decay('weights', kernel_shape, wd)
        biases = _variable_on_cpu('biases', [kernel_shape[-2]], tf.constant_initializer(0.0))
        deconv = _deconv2d(x, weights, biases, output_shape, kernel_stride)
    return deconv

def maxpool_2d_layer(x, kernel_shape, kernel_stride, layer_name):
    with tf.variable_scope(layer_name):
        max_pool = _max_pool(x, kernel_shape, kernel_stride, name)
    return max_pool
