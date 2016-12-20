import tensorflow as tf
import sys
import cv2
import os
import tensor_data
import numpy as np
import data_class
import resdeconv_model as model
import save_func as sf
import image_utility_func as iuf
#import save_func as sf

TRAIN_TXT = "../file_list/train_list5.txt"
TEST_TXT = "../file_list/test_list5.txt"
#TRAIN_TXT = "../file_list/spain_train_list1.txt"
#TEST_TXT = "../file_list/spain_test_list1.txt"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('feature_row',56,'''the feature row''')
tf.app.flags.DEFINE_string('feature_col',56,'''the feature col''')
tf.app.flags.DEFINE_string('feature_cha',1792,'''the feature channel''')


#tf.app.flags.DEFINE_string('label_row',299,'''the label row''')
#tf.app.flags.DEFINE_string('label_col',299,'''the label col''')
tf.app.flags.DEFINE_string('label_row',227,'''the label row''')
tf.app.flags.DEFINE_string('label_col',227,'''the label col''')
tf.app.flags.DEFINE_string('label_cha',1,'''the label channel''')

tf.app.flags.DEFINE_string('num_gpus',1,'''the number of gpu''')
tf.app.flags.DEFINE_string('batch_size',16, '''the batch size''')
tf.app.flags.DEFINE_string('restore_model',False,'''if restore the pre_trained_model''')

tf.app.flags.DEFINE_string('train_log_dir','resdeconv_train_log',
        '''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 1000000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate', 0.0001,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'resdeconv_models','''directory where to save the model''')
tf.app.flags.DEFINE_string('image_dir', 'tesdeconv_infer_images','''directory where to save the image''')
tf.app.flags.DEFINE_string('txt_log', 'train_log.txt','''directory where to save the display log''')

def norm_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image

def convert_image_name(image_name):
    image_name_list = image_name.split("/")
    last_two_name = image_name_list[-2] + "_" + image_name_list[-1]
    return last_two_name

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
    
def save_results(batch_label, batch_infer, batch_name, result_list):
    for i in xrange(len(batch_name)):
        image_name = batch_name[i].split(" ")[0].replace(".resnet_hypercolumn",".jpg")
        image_base_name = convert_image_name(image_name)
        image = iuf.load_image(image_name)
        label_norm = iuf.repeat_image(iuf.norm_image(batch_label[i]))
        infer_norm = iuf.repeat_image(iuf.norm_image(batch_infer[i]))
        stack_image = np.hstack((image, label_norm, infer_norm))
        iuf.show_image(stack_image, normalize = False)
        continue
        iuf.save_image(stack_image, FLAGS.result_dir + "/" + image_base_name.replace(".jpg", "resdeconv_result.jpg"))
        batch_infer[i].tofile(FLAGS.result_dir + "/" + image_base_name.replace(".jpg",".npy"))
        num_car_label = np.sum(batch_label[i])
        num_car_infer = np.sum(batch_infer[i])
        result_list.append(image_name + " " + str(num_car_label) + " " + str(num_car_infer))

def train():
    train_batch_data, train_batch_label, train_batch_name = gen_data_label(TRAIN_TXT, True)
    test_batch_data, test_batch_label, test_batch_name = gen_data_label(TEST_TXT, False)

    data_ph = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.feature_row,
                                                FLAGS.feature_col, FLAGS.feature_cha), name = 'feature')
    label_ph = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.label_row,
                                                FLAGS.label_col, FLAGS.label_cha), name = 'label')
    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    keep_prob_ph = tf.placeholder(tf.float32, name = 'keep_prob')
    train_test_phase_ph = tf.placeholder(tf.bool, name = 'phase_holder')

    output_shape = [FLAGS.batch_size, FLAGS.label_row, FLAGS.label_col, FLAGS.label_cha]

    #infer = model.test_infer_size(label_ph)

    #gpu_list = ['/gpu:0','/gpu:1','/gpu:2','/gpu:3']
    #for d in gpu_list:
    #    with tf.device(d):
    #        if (d != gpu_list[0]):
    #            tf.get_variable_scope().reuse_variables()
    infer, count_diff_infer = model.inference(data_ph, output_shape, keep_prob_ph, train_test_phase_ph)
    loss, _ = model.loss(infer, count_diff_infer, label_ph)
    train_op = model.train_op(loss, FLAGS.init_learning_rate, global_step)

    sess = tf.Session()

    sf.add_train_var()
    sf.add_loss()
    train_sum = tf.merge_all_summaries()

    #test_count_loss = tf.reduce_mean(count_diff_infer, name = "test_count")
    test_loss, test_count = model.loss(infer, count_diff_infer, label_ph)

    test_count_sum = tf.scalar_summary("test_count", tf.reduce_mean(test_count))
    test_sum = tf.scalar_summary("test_loss", test_loss)

    sum_writer = tf.train.SummaryWriter(FLAGS.train_log_dir, sess.graph)

    saver = tf.train.Saver()

    init_op = tf.initialize_all_variables()
    sess.run(init_op)


    if FLAGS.restore_model:
        sf.restore_model(sess, saver, FLAGS.model_dir)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)
    result_list = list()

    for i in xrange(FLAGS.max_training_iter):
        train_batch_data_v, train_batch_label_v, = sess.run([train_batch_data, train_batch_label])
        _, loss_v, infer_v, train_sum_v = sess.run([train_op, loss, infer, train_sum], {data_ph: train_batch_data_v, 
                                        label_ph: train_batch_label_v,
                                        train_test_phase_ph:True})
        if i % 20 == 0:
            test_batch_data_v, test_batch_label_v, test_batch_name_v = \
                                sess.run([test_batch_data, test_batch_label, test_batch_name])

            test_loss_v, test_count_v, infer_v ,test_sum_v ,test_count_sum_v = \
                                                                sess.run([test_loss, test_count,
                                                                        infer, test_sum, test_count_sum], 
                                                                        {data_ph:test_batch_data_v, 
                                                                        label_ph:test_batch_label_v,
                                                                        train_test_phase_ph:False})

            num_car_label = np.sum(test_batch_label_v)/FLAGS.batch_size
            num_car_infer = np.mean(test_count_v)
            num_car_diff = np.mean(np.abs(np.sum(test_batch_label_v, axis = (1,2,3)) - test_count_v))
            print("i: %d train_loss: %.5f, test_loss: %.5f, test_num_car: %.2f, infer_num_car: %.2f, num_car_diff: %.2f"%(i, 
                                loss_v, test_loss_v, num_car_label, num_car_infer, num_car_diff))
            
            sum_writer.add_summary(train_sum_v, i)
            sum_writer.add_summary(test_sum_v, i)
            sum_writer.add_summary(test_count_sum_v, i)
            #save_results(test_batch_label_v, infer_v, test_batch_name_v, result_list)
            #label_norm = iuf.norm_image(batch_label[i])

        if i != 0 and (i % 200 == 0 or i == FLAGS.max_training_iter - 1):
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
