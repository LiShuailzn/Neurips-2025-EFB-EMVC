import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from code.config import get_configs
opt = os.path

paras = get_configs()
image_size = paras['image_size']
w, h, c = image_size['w'], image_size['h'], image_size['c']


def get_image_paths(data_type='open_', base_dir='data', data_home='data'):
    if data_type == 'open_':
        fns = ['labelAll_open_train.txt', 'labelAll_open_test.txt']
    else:
        fns = ['labelAll_train.txt', 'labelAll_test.txt']
    train_fns, train_y = [], []
    test_fns, test_y = [], []
    with open(opt.join(base_dir, fns[0])) as f:
        for v in f.readlines():
            vv = v.strip().split(',')
            train_fns.append(opt.join(data_home, vv[1], vv[0].split('_')[-1]))
            train_y.append(int(vv[1]))
    with open(opt.join(base_dir, fns[1])) as f:
        for v in f.readlines():
            vv = v.strip().split(',')
            test_fns.append(opt.join(data_home, vv[1], vv[0].split('_')[-1]))
            test_y.append(int(vv[1]))
    return train_fns, train_y, test_fns, test_y


def read_image(fn):
    img = Image.open(fn)
    img = img.convert('L')
    img = img.resize((w, h))
    img = np.array(img)
    return np.array(img)


def get_checkAll(data_type='open_'):
    train_X_npy = opt.join(data_type+'train_X1.npy')
    train_Y_npy = opt.join(data_type+'train_Y1.npy')
    test_X_npy = opt.join(data_type+'test_X1.npy')
    test_Y_npy = opt.join(data_type+'test_Y1.npy')
    base_dir = opt.join('data_utils', 'fn')
    data_home = opt.join('..', 'data', 'checkAll')
    train_fns, train_y, test_fns, test_y = get_image_paths(
        data_type=data_type, base_dir=base_dir, data_home=data_home)
    if data_type == 'open_':
        train_Y = np.array(train_y) - 10000
        test_Y = np.array(test_y) - 10000
    else:
        train_Y = np.array(train_y)
        test_Y = np.array(test_y)
    np.save(test_Y_npy, test_Y)
    np.save(train_Y_npy, train_Y)
    train_X, test_X = [], []
    for i in tqdm(train_fns):
        train_X.append(read_image(i))
    train_X = np.array(train_X)
    np.save(train_X_npy, train_X)
    for i in tqdm(test_fns):
        test_X.append(read_image(i))
    test_X = np.array(test_X)
    np.save(test_X_npy, test_X)

if __name__ == '__main__':
    get_checkAll(data_type='')



