import tensorflow as tf
import sys
import cv2
import os
import tensor_data
import numpy as np
import data_class
import fcn_model
import save_func as sf
#import save_func as sf

TRAIN_TXT = "../file_list/train_list2.txt"
TEST_TXT = "../file_list/test_list2.txt"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('feature_row',8,'''the feature row''')
tf.app.flags.DEFINE_string('feature_col',8,'''the feature col''')
tf.app.flags.DEFINE_string('feature_cha',2048,'''the feature channel''')


tf.app.flags.DEFINE_string('label_row',299,'''the label row''')
tf.app.flags.DEFINE_string('label_col',299,'''the label col''')
tf.app.flags.DEFINE_string('label_cha',1,'''the label channel''')

tf.app.flags.DEFINE_string('num_gpus',1,'''the number of gpu''')
tf.app.flags.DEFINE_string('batch_size',32,'''the batch size''')
tf.app.flags.DEFINE_string('restore_model',False,'''if restore the pre_trained_model''')

tf.app.flags.DEFINE_string('train_log_dir','train_log',
        '''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 1000000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate', 0.0001,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'models','''directory where to save the model''')
tf.app.flags.DEFINE_string('image_dir', 'infer_images','''directory where to save the image''')
tf.app.flags.DEFINE_string('txt_log', 'train_log.txt','''directory where to save the display log''')

def norm_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image
    
def gen_data_label(file_name, is_train):
    input_class = data_class.DataClass(tf.constant([], tf.string))
    input_class.decode_class= data_class.BINClass([FLAGS.feature_row, FLAGS.feature_col, FLAGS.feature_cha])
    
    label_class = data_class.DataClass(tf.constant([], tf.string))
    label_class.decode_class = data_class.BINClass([FLAGS.label_row, FLAGS.label_col, FLAGS.label_cha])
    
    tensor_list = [input_class] + [label_class]


    file_queue = tensor_data.file_queue(file_name, is_train)
    batch_tensor_list = tensor_data.file_queue_to_batch_data(file_queue,
            tensor_list, is_train, FLAGS.batch_size)

    return batch_tensor_list

def train():
    train_batch_data, train_batch_label, _ = gen_data_label(TRAIN_TXT, True)
    test_batch_data, test_batch_label, _ = gen_data_label(TEST_TXT, False)

    data_ph = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.feature_row,
                                                FLAGS.feature_col, FLAGS.feature_cha), name = 'feature')
    label_ph = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.label_row,
                                                FLAGS.label_col, FLAGS.label_cha), name = 'label')
    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    keep_prob_ph = tf.placeholder(tf.float32, name = 'keep_prob')
    train_test_phase_ph = tf.placeholder(tf.bool, name = 'phase_holder')

    #fcn_model.test_infer_size(label_ph)
    output_shape = [FLAGS.batch_size, FLAGS.label_row, FLAGS.label_col, FLAGS.label_cha]

    infer = fcn_model.inference(data_ph, output_shape, keep_prob_ph, train_test_phase_ph)
    loss = fcn_model.loss(infer, label_ph)
    train_op = fcn_model.train_op(loss, FLAGS.init_learning_rate, global_step)

    test_loss = fcn_model.loss(infer, label_ph)
    sess = tf.Session()

    saver = tf.train.Saver()

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    for i in xrange(FLAGS.max_training_iter):
        train_batch_data_v, train_batch_label_v = sess.run([train_batch_data, train_batch_label])
        _, loss_v, infer_v = sess.run([train_op, loss, infer], {data_ph: train_batch_data_v, label_ph: train_batch_label_v})

        if i % 500 == 0:
            test_batch_data_v, test_batch_label_v = sess.run([test_batch_data, test_batch_label])
            test_loss_v, infer_v = sess.run([test_loss, infer], {data_ph:test_batch_data_v, label_ph:test_batch_label_v})
            num_car_label = np.sum(test_batch_label_v)/FLAGS.batch_size
            num_car_infer = np.sum(infer_v)/FLAGS.batch_size
            print("i: %d train_loss: %.5f, test_loss: %.5f, test_num_car: %.2f, infer_num_car: %.2f"%(i, 
                                loss_v, test_loss_v, num_car_label, num_car_infer))

        if i % 5000 == 0 or i == FLAGS.max_training_iter - 1:
            sf.save_model(sess, saver, FLAGS.model_dir, i)
            #label_norm = norm_image(test_batch_label_v[0])
            #infer_norm = norm_image(infer_v[0])
            #image = np.hstack((label_norm, infer_norm))
            #cv2.imwrite(FLAGS.image_dir + "/%08d.jpg"%(i/100), image)
            #cv2.imshow("infer", image)

def main(argv = sys.argv):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    train()

if __name__ == "__main__":
    tf.app.run()
