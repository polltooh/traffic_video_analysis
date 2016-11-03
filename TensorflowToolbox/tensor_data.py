import tensorflow as tf
import data_class

def check_list(data):
    """check if data is a list, if it is not a list, it will return a list as [data]"""
    if type(data) is not list:
        return [data]
    else:
        return data

def file_queue(file_name, shuffle_data):
    """ data_file_name_queue
    Args:
            file_name: txt file, file name or a list of file name
            shuffle: True or False
    """
    if type(file_name) is not list:
            file_name = [file_name]
            
    data_filenamequeue = tf.train.string_input_producer(file_name, shuffle=shuffle_data)
    return data_filenamequeue	


def file_queue_to_batch_data(filename_queue, data_classes, is_train, batch_size, 
				whiten_data = True, distort_data = True):
    """ batch data
    Args:
            filename_queue: produced by file_queue
            data_classes: list of DataClass
            is_train: True or False
            batch_size: batch size
            whiten_data: apply image whitening to the image
            distort_data: apply random cropping and fliping 
    """
    if is_train:
       distort_data = False

    data_classes = check_list(data_classes)
    data_type_list = list()
    for data in data_classes:
            data_type_list.append(data.data_format)
    
    line_reader = tf.TextLineReader()
    key, next_line = line_reader.read(filename_queue)
    data_list = tf.decode_csv(next_line, data_type_list, field_delim=" ")
    tensor_list = list()
    for i in xrange(len(data_classes)):
            if (data_classes[i].decode_class is not None):
                    tensor_list.append(data_classes[i].decode_class.decode(data_list[i], distort_data,
                                                    whiten_data))
            else:
                    tensor_list.append(data_list[i])
    tensor_list.append(next_line)

    if is_train:
            min_after_dequeue = 100
            capacity = min_after_dequeue + (1 + 2) * batch_size
            batch_tensor_list = tf.train.shuffle_batch(
                            tensor_list, batch_size = batch_size, capacity=capacity,
                            min_after_dequeue=min_after_dequeue, num_threads = 2)
    else:
            batch_tensor_list = tf.train.batch(
                            tensor_list, batch_size=batch_size, num_threads = 2)

    return batch_tensor_list
