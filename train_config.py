import datetime
import os

image_size = 128
image_channel = 3
epoch_size =200
train_rate = 0.8
batch_size = 13
val_step = 2
learning_rate = 1e-3
is_train = True
is_show = not is_train
is_none_load_model = False and is_train
steps_per_epoch = 25

device = '/gpu:0'
project_path = './'
#   模型改动后要重新命名
model_name = ('benign_dataseta')   # 关键！修改名称会重新创建模型 否则会默认加载已经保存过的模型
train_dataset_path = project_path + "archive/Dataset_BUSI_with_GT/benign/train_dataset"
val_dataset_path = project_path + "archive/Dataset_BUSI_with_GT/benign/valid_dataset"
__save_path = project_path + "Save_Data/"
image_path = __save_path + f"images/{model_name}/"

try:
    os.makedirs(image_path)
except OSError:
    print('dir has exists')

log_path = __save_path + f"logs/{model_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
model_path = __save_path + f"models/{model_name}/"

if __name__ == '__main__':
    import tensorflow as tf

    print(tf.config.list_physical_devices())