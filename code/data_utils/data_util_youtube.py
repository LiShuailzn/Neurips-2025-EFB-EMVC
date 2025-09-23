import tensorflow as tf
import numpy as np
import os
from code import config

opt = os.path
paras = config.get_configs()
nb_view = paras['nb_view']
image_size = paras['image_size']
w, h, c = image_size['w'], image_size['h'], image_size['c']

def get_data(data_base_dir='..'):
    print('Data loading ......')
    train_x = np.load(os.path.join(data_base_dir, 'train_X.npy'))
    test_x = np.load(os.path.join(data_base_dir, 'test_X.npy'))
    if c == 1:
        train_x = np.expand_dims(train_x, axis=-1)
        test_x = np.expand_dims(test_x, axis=-1)
    train_x = (train_x / 127.5) - 1.
    test_x = (test_x / 127.5) - 1.
    train_y = np.load(os.path.join(data_base_dir, 'train_Y.npy'))
    test_y = np.load(os.path.join(data_base_dir, 'test_Y.npy'))
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    print('Data loading finished！！！')
    return train_x, train_y, test_x, test_y



#  youtube
def get_views(view_data_dir='/home/lishuai_lxy/fph/YoutubeFace'):
    num_views = 5
    view_train_x = []
    view_test_x = []
    for i in range(1, num_views + 1):
        train_file = os.path.join(view_data_dir, f'train_{i}.npy')
        test_file = os.path.join(view_data_dir, f'test_{i}.npy')
        view_train_x.append(np.load(train_file))
        view_test_x.append(np.load(test_file))
        print(f"Loading view_train and view_test for view {i}")
    train_y_file = os.path.join(view_data_dir, 'train_y.npy')
    test_y_file = os.path.join(view_data_dir, 'test_y.npy')
    train_y = np.load(train_y_file)
    test_y = np.load(test_y_file)
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    # 打印 view_train_x 中每个元素的维度
    print(f"Dimensions of view_train_x: {[arr.shape for arr in view_train_x]}")
    # 打印标签的维度
    print(f"Dimensions of train_y: {train_y.shape}")
    print(f"Dimensions of test_y: {test_y.shape}")

    return view_train_x, train_y, view_test_x, test_y

def load_teacher_labels(view_data_dir='views', models_ls=None):
    models_ls = ['v0', 'v1', 'v2', 'v3', 'v4']

    teacher_labels = []  # 用来存储软标签

    for i, model in enumerate(models_ls):
        # 软标签文件路径
        soft_label_path = os.path.join(view_data_dir, f'view_{i}_train_Y.npy')

        # 检查软标签文件是否存在
        if os.path.exists(soft_label_path):
            print(f"Loading soft label for model {model} from {soft_label_path}")
            # 加载软标签
            soft_label = np.load(soft_label_path)
            teacher_labels.append(soft_label)
        else:
            print(f"Warning: Soft label file for model {model} (view {i}) not found at {soft_label_path}.")
            teacher_labels.append(None)  # 如果找不到软标签文件，设置为 None

    return teacher_labels

def load_teacher_logits(view_data_dir='views', models_ls=None):
    models_ls = ['v0', 'v1', 'v2', 'v3', 'v4']

    teacher_labels = []  # 用来存储软标签

    for i, model in enumerate(models_ls):
        # 软标签文件路径
        soft_label_path = os.path.join(view_data_dir, f'view_{i}_logits.npy')

        # 检查软标签文件是否存在
        if os.path.exists(soft_label_path):
            print(f"Loading soft label for model {model} from {soft_label_path}")
            # 加载软标签
            soft_label = np.load(soft_label_path)
            teacher_labels.append(soft_label)
        else:
            print(f"Warning: Soft label file for model {model} (view {i}) not found at {soft_label_path}.")
            teacher_labels.append(None)  # 如果找不到软标签文件，设置为 None

    return teacher_labels


def load_cost_matrices(view_data_dir='views', models_ls=None):
    models_ls = ['v0', 'v1', 'v2', 'v3', 'v4']

    cost_matrices = []  # 用来存储成本矩阵

    for i, model in enumerate(models_ls):
        # 成本矩阵文件路径
        cost_matrix_path = os.path.join(view_data_dir, f'view_{i}_kernel', 'cost_matrix.npy')

        # 检查成本矩阵文件是否存在
        if os.path.exists(cost_matrix_path):
            print(f"Loading cost matrix for model {model} from {cost_matrix_path}")
            # 加载成本矩阵
            cost_matrix = np.load(cost_matrix_path)
            cost_matrices.append(cost_matrix)
        else:
            print(f"Warning: Cost matrix file for model {model} (view {i}) not found at {cost_matrix_path}.")
            cost_matrices.append(None)  # 如果找不到成本矩阵文件，设置为 None

    return cost_matrices