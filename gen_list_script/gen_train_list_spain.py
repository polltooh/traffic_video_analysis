import file_io

train_file_dir = "../data/toJeff/Train/"
test_file_dir = "../data/toJeff/Test/"


def get_list_file(file_dir):
    #train_feature_list = file_io.get_listfile(file_dir + "Resized_images/","resnet_hypercolumn")
    train_feature_list = file_io.get_listfile(file_dir + "Resized_images/","jpg")
    train_label = file_io.get_listfile(file_dir + "Resized_GTdensity/","npy")
    train_list = list()

    for tf in train_feature_list:
        #tf_label = tf.replace(".resnet_hypercolumn","dots.png.npy").replace("images","GTdensity")
        tf_label = tf.replace(".jpg","dots.png.npy").replace("images","GTdensity")
        assert(tf_label in train_label)
        train_list.append(tf + " " + tf_label)
    return train_list

train_list = get_list_file(train_file_dir)
test_list = get_list_file(test_file_dir)
print(len(train_list))
print(len(test_list))

file_io.save_file(train_list, "../file_list/spain_train_list2.txt", True)
file_io.save_file(test_list, "../file_list/spain_test_list2.txt")
