import tensorflow as tf
import sys
import cv2
import os
import tensor_data
import numpy as np
import data_class
import scipy.stats as ss
import resdeconv_model as model
import save_func as sf
import file_io
import image_utility_func as iuf
#import save_func as sf

#TEST_TXT = "../file_list/test_list5.txt"
TEST_TXT = "../file_list/cam691_test1.txt"
#TEST_TXT = "../file_list/spain_test_list1.txt"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('feature_row',56,'''the feature row''')
tf.app.flags.DEFINE_string('feature_col',56,'''the feature col''')
tf.app.flags.DEFINE_string('feature_cha',1792,'''the feature channel''')

tf.app.flags.DEFINE_string('label_row',227,'''the label row''')
tf.app.flags.DEFINE_string('label_col',227,'''the label col''')
tf.app.flags.DEFINE_string('label_cha',1,'''the label channel''')

tf.app.flags.DEFINE_string('num_gpus',1,'''the number of gpu''')
tf.app.flags.DEFINE_string('batch_size',16,'''the batch size''')
tf.app.flags.DEFINE_string('restore_model',False,'''if restore the pre_trained_model''')

tf.app.flags.DEFINE_string('train_log_dir','train_log',
        '''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 100000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate', 0.0001,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'resdeconv_models','''directory where to save the model''')
tf.app.flags.DEFINE_string('result_dir', 'cam691_results','''directory where to save the results''')
tf.app.flags.DEFINE_string('txt_log', 'train_log.txt','''directory where to save the display log''')

def convert_image_name(image_name):
    image_name_list = image_name.split("/")
    last_two_name = image_name_list[-2] + "_" + image_name_list[-1]
    return last_two_name


def save_results(batch_label, batch_infer, batch_diff_infer, batch_name, result_list):
    for i in xrange(len(batch_name)):
        image_name = batch_name[i].split(" ")[0].replace(".resnet_hypercolumn",".jpg")
        image_base_name = convert_image_name(image_name)
        image = iuf.load_image(image_name)
        image = iuf.resize_image(image, (FLAGS.label_row, FLAGS.label_col))
        label_norm = iuf.repeat_image(iuf.norm_image(batch_label[i]))

        num_car_label = np.sum(batch_label[i])
        num_car_infer = np.sum(batch_infer[i]) + batch_diff_infer[i][0]

        batch_infer[i] = ss.threshold(batch_infer[i], threshmin = 0.0, newval = 0)
        infer_norm = iuf.repeat_image(iuf.norm_image(batch_infer[i]))
        stack_image = np.hstack((image, label_norm, infer_norm))
        #iuf.show_image(stack_image, normalize = False)
        iuf.save_image(stack_image, FLAGS.result_dir + "/" + image_base_name.replace(".jpg", "resdeconv_result.jpg"))
        batch_infer[i].tofile(FLAGS.result_dir + "/" + image_base_name.replace(".jpg",".npy"))
        print("label: %.2f, infer: %.2f"%(num_car_label, num_car_infer))
        result_list.append(image_name + " " + str(num_car_label) + " " + str(num_car_infer))

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
    print(file_io.get_file_length(TEST_TXT)/FLAGS.batch_size)
    FLAGS.max_training_iter = file_io.get_file_length(TEST_TXT)/FLAGS.batch_size
    test_batch_data, test_batch_label, test_batch_name = gen_data_label(TEST_TXT, False)


    data_ph = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.feature_row,
                                                FLAGS.feature_col, FLAGS.feature_cha), name = 'feature')
    label_ph = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.label_row,
                                                FLAGS.label_col, FLAGS.label_cha), name = 'label')
    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    keep_prob_ph = tf.placeholder(tf.float32, name = 'keep_prob')
    train_test_phase_ph = tf.placeholder(tf.bool, name = 'phase_holder')

    #fcn_model.test_infer_size(label_ph)
    output_shape = [FLAGS.batch_size, FLAGS.label_row, FLAGS.label_col, FLAGS.label_cha]

    infer, diff_infer = model.inference(data_ph, output_shape, keep_prob_ph, train_test_phase_ph)
    #loss = model.loss(infer, diff_infer, label_ph)
    #train_op = model.train_op(loss, FLAGS.init_learning_rate, global_step)

    #test_loss = model.loss(infer, diff_infer, label_ph)
    sess = tf.Session()

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    saver = tf.train.Saver()
    sf.restore_model(sess, saver, FLAGS.model_dir, model_name = None)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)
    result_list = list()
    for i in xrange(FLAGS.max_training_iter):
        #test_batch_name_v = sess.run(test_batch_name)
        test_batch_data_v, test_batch_label_v, test_batch_name_v = sess.run([test_batch_data, test_batch_label, test_batch_name])
        #test_loss_v, infer_v = sess.run([test_loss, infer], {data_ph:test_batch_data_v, label_ph:test_batch_label_v})
        diff_infer_v, infer_v = sess.run([diff_infer, infer], {data_ph:test_batch_data_v, label_ph:test_batch_label_v})
        save_results(test_batch_label_v, infer_v, diff_infer_v, test_batch_name_v, result_list)

    file_io.save_file(result_list, FLAGS.result_dir + "/results.txt")

def main(argv = sys.argv):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    train()

if __name__ == "__main__":
    tf.app.run()
