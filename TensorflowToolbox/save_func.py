import tensorflow as tf
import time


FLAGS = tf.app.flags.FLAGS

def add_train_var():
    """ add all trainable variable to summary"""
    for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

def add_loss(loss_scope = 'losses'):
    """ add all losses to summary """
    for l in tf.get_collection(loss_scope):
            tf.scalar_summary(l.op.name, l)

def restore_model(sess, saver, model_dir, model_name = None):
    """ restore model:
            if model_name is None, restore the last one
    """
    if model_name is None:
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt and ckpt.all_model_checkpoint_paths[-1]:
                    print("restore " + ckpt.all_model_checkpoint_paths[-1])
                    saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            else:
                    print('no check point')
    else:
            print("restore " + model_name)
            saver.restore(sess, model_dir + '/' + model_name)
	
def save_model(sess, saver, model_dir, iteration):
    """ save the current model"""

    curr_time = time.strftime("%Y%m%d_%H%M")
    model_name = model_dir + '/' + curr_time + \
                            '_iter_' + str(iteration) + '_model.ckpt'
    saver.save(sess, model_name)

def add_value_sum(summary_writer, value, name):
    """ add python value to tensorboard """
    return tf.Summary(value = [tf.Summary.Value(tag = name, simple_value = value)])	

def group_mv_ops(train_op, moving_average_decay, global_step):
    """ group all the operations 
    Args:
    """	
    # batchnorm_updates = tf.get_collection(FLAGS.bn_collection)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # batchnorm_vars = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)	
    variables_to_average = tf.trainable_variables()
                                                    # batchnorm_vars)
            
    variables_averages_op = variable_averages.apply(variables_to_average)
    # batchnorm_updates_op = tf.group(*batchnorm_updates)
    all_op = tf.group(train_op, variables_averages_op)

    return all_op
