import tensorflow as tf
import os
from code.config import get_configs
from .data_utils import get_image_paths
opt = os.path

paras = get_configs()
AUTOTUNE = tf.data.experimental.AUTOTUNE

w, h, c = paras['image_size']['w'], paras['image_size']['h'], paras['image_size']['c']


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [w, h])
    image /= 255.0
    return image


def load_and_preprocess_image(img_path):
    image = tf.io.read_file(img_path)
    return preprocess_image(image)


def get_imgs_path(base_dir):
    image_paths = []
    labels = []
    for label, v in enumerate(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, v)
        for vv in os.listdir(subdir):
            img_path = os.path.join(subdir, vv)
            image_paths.append(img_path)
            labels.append(label)
    return image_paths, labels


def save_tfrecodes(image_paths, record_file='images_tfrec'):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    ds = image_ds.map(tf.io.serialize_tensor)
    tfrec = tf.data.experimental.TFRecordWriter(record_file)
    tfrec.write(ds)


def load_data(labels, record_file='images_tfrec'):
    ds = tf.data.TFRecordDataset(record_file)

    def parse(x):
        result = tf.io.parse_tensor(x, out_type=tf.float32)
        result = tf.reshape(result, [w, h, c])
        return result
    ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.case(labels), tf.int64)
    image_label_ds = tf.data.Dataset.zip((ds, label_ds))
    # image_label_ds = image_label_ds.cache(filename='./cache.tf-data')
    ds = image_label_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=paras['image_count'])
    )
    ds = ds.batch(paras['batch_size'])
    ds.prefetch(buffer_size=AUTOTUNE)
    return ds


if __name__ == '__main__':
    base_dir = os.path.join('..', 'data', 'checkPart')
    base_dir = opt.join('..', 'data')
    train_fns, train_y, test_fns, test_y = get_image_paths(base_dir=base_dir)
    train_fns = [opt.join('data', v) for v in train_fns]
    save_tfrecodes(base_dir=base_dir, record_file='images_tfrec')
