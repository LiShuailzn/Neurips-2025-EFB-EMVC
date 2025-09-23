import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from os.path import join as pjoin
from multiprocessing import Pool
import time
from code.data_utils.data_util_youtube import get_views
from code.data_utils.data_util_youtube import load_teacher_logits
from code.data_utils.data_util_youtube import load_cost_matrices
from code import gen_offspring_tree, population_init_tree, utils, config
from code.code2net_tree_nus_ntu_ego import code2net_tree
from code.new_loss import wkd_logit_loss_with_separation


paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = paras['fused_nb_feats']
classes = paras['classes']
batch_size = paras['batch_size']
epochs = paras['epochs']
classes = paras['classes']
pop_size = paras['pop_size']
nb_iters = paras['nb_iters']
data_name = paras['data_name']

data_base_dir = os.path.join('..', data_name)
view_data_dir = os.path.join(data_base_dir, 'view')
view_train_x, train_y, view_test_x, test_y = get_views(view_data_dir='/mnt/disk1/lishuai/EA-Dataset/YoutubeFace/test_1')
print("Shape of train_y:", train_y.shape)  # 确保 train_y 的形状正确
teacher_logits = load_teacher_logits(view_data_dir='/mnt/disk1/lishuai/EA-Dataset/YoutubeFace/test_1/teacher_logits')
cost_matrices = load_cost_matrices(view_data_dir='/mnt/disk1/lishuai/EA-Dataset/YoutubeFace/test_1')
def train_individual(individual_code, result_save_dir='.', gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(individual_code)
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    view_train_xx, view_test_xx = [], []
    for code in individual_code:
        if code[0] != '-':
            index = int(code[0])
            view_train_xx.append(view_train_x[index])
            view_test_xx.append(view_test_x[index])
    individual_code_str = '+'.join([str(ind) for ind in individual_code])
    nb_feats = [i.shape[1] for i in view_train_x]
    model = code2net_tree(individual_code=[], nb_feats=nb_feats,listtree=individual_code)
    num_views = len(view_train_xx)
    loss_weights = {
        'fusion_output': 1.0 * num_views,
    }
    i=0
    for code in individual_code:
        if code[0] != '-':
            index = int(code[0])
            loss_weights[f'view_{index}_output_{i}'] = 1.0
            i=i+1
    loss = {
        'fusion_output': 'categorical_crossentropy',
    }
    j = 0
    for code in individual_code:
        if code[0] != '-':
            index = int(code[0])
            loss[f'view_{index}_output_{j}'] = lambda y_true, y_pred: wkd_logit_loss_with_separation(
                logits_student=y_pred,
                logits_teacher=teacher_logits[index],
                gt_label=y_true,
                temperature=2,
                gamma=0.3,
                cost_matrix=cost_matrices[index],
                sinkhorn_lambda=0.05,
                sinkhorn_iter=10
            )
            j += 1
    adam = tf.keras.optimizers.Adam()
    model.compile(
        loss=loss,
        loss_weights=loss_weights,
        optimizer=adam,
        metrics={
            'fusion_output': ['accuracy'],
        }
    )
    print("Loss weights:", loss_weights)
    checkpoint_filepath = os.path.join(result_save_dir, individual_code_str + '.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_fusion_output_accuracy', verbose=0, save_best_only=True,
        save_weights_only=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_fusion_output_accuracy', patience=paras['patience'])
    csv_filepath = os.path.join(result_save_dir, individual_code_str + '.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_filepath)
    train_labels = {'fusion_output': train_y}
    k = 0
    for code in individual_code:
        if code[0] != '-':
            index = int(code[0])
            train_labels[f'view_{index}_output_{k}'] = train_y
            k += 1
    model.fit(view_train_xx, train_labels, batch_size=batch_size, epochs=epochs,
              verbose=0, validation_data=(view_test_xx, test_y),
              callbacks=[csv_logger, early_stop, checkpoint])
    result_df = pd.read_csv(csv_filepath)
    max_f1_score = result_df['val_fusion_output_accuracy'].max()
    return individual_code_str + ',' + str(max_f1_score)

def find_same_code_acc(individual_code, result_save_dir='.'):
    individual_code_str = '-'.join([str(ind) for ind in individual_code])
    return individual_code_str

def record_code(individual_code, result_save_dir='.'):
    individual_code_str = '-'.join([str(ind) for ind in individual_code])
    return individual_code_str


def list2str(list1):
    return '-'.join([str(i) for i in list1])

def list2str_tree(list1):
    return '+'.join([str(i) for i in list1])

def multi_proccess_train(i_iter, Q_t, shared_code_sets):
    gpu_list = paras['gpu_list']
    gpus = len(gpu_list)
    gpu_idx = 0
    pool = Pool(gpus)
    individual_code_str = []
    pop_size1 = len(Q_t)
    for ind_i in np.arange(0, pop_size1):
        print(len(Q_t), '==========', ind_i+1)
        code_str = list2str_tree(Q_t[ind_i])
        utils.write_result_file(','.join([str(i_iter + 1), code_str]),
                                fn=os.path.join(result_save_dir, 'history.csv'))
        if code_str not in shared_code_sets:
            shared_code_sets.add(code_str)
            individual_code_str.append(
                pool.apply_async(func=train_individual, args=(Q_t[ind_i], result_save_dir, str(gpu_list[gpu_idx])))
            )
            gpu_idx += 1

        if gpu_idx == gpus or ind_i == (pop_size1-1):
            pool.close()
            pool.join()
            for ss in individual_code_str:
                utils.write_result_file(ss.get(), fn=os.path.join(result_save_dir, 'result.csv'))
            pool = Pool(gpus)
            gpu_idx = 0
            individual_code_str = []


def train():
    shared_code_sets = set()
    # 1. population initialization
    print(f'The number of views: {len(view_train_x)}')
    ini_population = population_init_tree.generate_population_tree(
        views=len(view_train_x), pop_size=pop_size, verbose=0)

    # 2.init population fitness
    start = time.time()
    P_t = ini_population
    multi_proccess_train(i_iter=-1, Q_t=P_t, shared_code_sets=shared_code_sets)

    # 3. gen_offspring
    for i in tqdm(range(paras['nb_iters'])):
        print(f'==================={i+1}/', paras['nb_iters'])
        Q_t = gen_offspring_tree.gen_offspring(P_t)
        multi_proccess_train(i_iter=i, Q_t=Q_t, shared_code_sets=shared_code_sets)
        P_t = gen_offspring_tree.selection(P_t, Q_t)
        print('=' * 60, i+1, 'End.')

    print(f'Total time is :{time.time()-start}')
    utils.write_result_file(str(time.time() - start), fn=os.path.join(result_save_dir, 'history.csv'))

if __name__ == '__main__':
    result_save_dir = pjoin(data_name+'_view_result', paras['result_save_dir'])
    print(result_save_dir)
    print(data_name, fused_nb_feats)
    os.makedirs(result_save_dir, exist_ok=True)
    train()
    print(result_save_dir)
    print(data_name, fused_nb_feats)