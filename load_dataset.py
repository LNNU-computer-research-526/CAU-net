import os.path
import random
from train_config import *
from Data_augement import data_augment
import tensorflow as tf



def parse_image(path):
    mask = tf.io.read_file(path)
    mask = mask_deal(mask)

    def _has_path(p):
        p = p.numpy()
        return os.path.exists(p)

    if tf.reduce_sum(mask) == 0:
        k_count = 0
    else:
        k_count = 1
    for j in [1, 2, 3]:
        path_j = tf.strings.regex_replace(path, "_mask.png", f"_mask_{j}.png")
        path_j = tf.py_function(_has_path, [path_j], Tout=tf.bool)
        if path_j:
            k_count += 1
            mask_j = tf.io.read_file(tf.strings.regex_replace(path, "_mask.png", f"_mask_{j}.png"))
            mask_j = mask_deal(mask_j)
            mask = mask_j + mask
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    image = tf.io.read_file(tf.strings.regex_replace(path, "_mask", ""))
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size])

    return image, mask, k_count


def mask_deal(mask):
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, [image_size, image_size])
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return mask


def data_aug(image, mask, k_count):
    out = tf.numpy_function(data_augment, [image, mask], [tf.float32, tf.float32])
    return out[0], out[1], k_count

def get_files_dataset(_dir):
    files = list(tf.io.gfile.glob(str(_dir) + "/*mask.png"))
    random.shuffle(files)
    str_path = [str(path) for path in files]
    dataset = tf.data.Dataset.from_tensor_slices(str_path)
    dataset = dataset.map(parse_image)
    return dataset


train_dataset = (get_files_dataset(train_dataset_path)
                 .map(data_aug)
                 .map(lambda x, y, z: ((x, y, z), None))
                 .batch(batch_size).repeat(epoch_size))
test_dataset = (get_files_dataset(val_dataset_path)
                .map(lambda x, y, z: ((x, y, z), None))
                .batch(batch_size).repeat(epoch_size))
plot_dataset = (get_files_dataset(val_dataset_path)
                .map(lambda x, y, z: ((x, y, z), None))
                .batch(batch_size))





