import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from code.data_utils.data_util_youtube import get_views
from code.decode_T import code2net_separate


view_train_x, train_y, view_test_x, test_y = get_views(view_data_dir='/mnt/disk1/lishuai/EA-Dataset/YoutubeFace/test_1')

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, view_data, labels, batch_size, shuffle=True):
        self.view_data = view_data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(view_data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.view_data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        # 计算当前批次的数据范围
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.view_data[i] for i in batch_indices]
        batch_labels = self.labels[batch_indices]
        return np.array(batch_data), np.array(batch_labels)


def train_teacher(individual_code, result_save_dir='.', gpu='3'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # 视图个数和训练数据准备
    nb_view = (len(individual_code) + 1) // 2
    view_train_xx, view_test_xx = [], []
    for i in individual_code[:nb_view]:
        view_train_xx.append(view_train_x[i])
        view_test_xx.append(view_test_x[i])
    view_names = [f'view_{str(ind)}' for ind in individual_code[:nb_view]]
    nb_feats = [i.shape[1] for i in view_train_x]

    # 为每个视图生成单独的模型
    models = code2net_separate(individual_code=individual_code, nb_feats=nb_feats)
    results = []  # 保存训练结果

    for i, view_name in enumerate(view_names):
        print(f"Shape of view_train_xx[{i}]: {np.shape(view_test_xx[i])}")
        # 定义保存路径和日志路径
        checkpoint_filepath = os.path.join(result_save_dir, f'{view_name}_best.h5')
        csv_filepath = os.path.join(result_save_dir, f'{view_name}_training_log.csv')

        # 编译每个视图的模型
        model = models[i]
        adam = tf.keras.optimizers.Adam()
        topk = tf.keras.metrics.top_k_categorical_accuracy
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', topk])

        # 设置回调
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)
        csv_logger = tf.keras.callbacks.CSVLogger(csv_filepath)

        # 创建数据生成器用于训练
        train_generator = DataGenerator(view_train_xx[i], train_y, batch_size=512)
        val_generator = DataGenerator(view_test_xx[i], test_y, batch_size=512)

        # 训练模型
        model.fit(train_generator, epochs=200, verbose=0, validation_data=val_generator,
                  callbacks=[csv_logger, early_stop, checkpoint])

    return 0


# 测试
individual_code = [0,1,2,3,4,
                   0,0,0,0,]

result_save_dir='/mnt/disk1/lishuai/NIPS/teacher_youtube/test_1'
gpu = '7'  # 指定 GPU
train_teacher(individual_code, result_save_dir, gpu)