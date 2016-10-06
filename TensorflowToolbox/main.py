import tensorflow as tf
import sys
import os
import dataqueue
from train import train

file_name = "../src/build/data/guanhanw/combined.data"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('num_gpus',1,'''the number of gpu''')
tf.app.flags.DEFINE_string('batch_size',20,'''the batch size''')
tf.app.flags.DEFINE_string('feature_dim',10,'''the feature dimension''')
tf.app.flags.DEFINE_string('label_dim',9,'''the label dimension''')

tf.app.flags.DEFINE_string('train_log_dir','visual_audio_logs',
        '''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 10000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.1,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'models','''directory where to save the model''')
tf.app.flags.DEFINE_string('txt_log', 'train_log.txt','''directory where to save the display log''')


def main(argv = sys.argv):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)

    dataqueue = dataqueue.dataqueue(file_name)
    train(FLAGS, dataqueue)

if __name__ == '__main__':
    tf.app.run()

