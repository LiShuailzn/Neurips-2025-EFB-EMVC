import tensorflow as tf
import os
from code import config

paras = config.get_configs()
data_name = paras['data_name']
idxx = 0
def get_nb_view_by_individal_code(code):
    nb_view = (len(code) + 1) // 2  # 视图个数
    return nb_view


def write_result_file(str, fn='result.csv'):
    with open(fn, 'a+') as f:
        f.write(str)
        f.write('\n')
        f.flush()

def load_result(result_fn=os.path.join(data_name+'_view_result', paras['result_save_dir'], 'result.csv')):
    shared_code_acc = {}
    shared_code_acc_set = set()
    with open(result_fn) as f:
        for item in f.readlines():
            items = item.strip().split(',')
            if items[0] not in shared_code_acc_set:
                shared_code_acc[items[0]] = float(items[1])
    return shared_code_acc


def list2str(list1):
    return '-'.join([str(i) for i in list1])

def sign_sqrt(x):
    # mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-10)

def l2_norm(x):
    return tf.keras.backend.l2_normalize(x, axis=-1)