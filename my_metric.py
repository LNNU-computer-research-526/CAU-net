import keras.layers
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score, recall_score
from keras import backend as K
import keras
import matplotlib.pyplot as plt
from keras.callbacks import Callback

from load_dataset import *
from train_config import image_path



def _dice(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def _jaccard(y_true, y_pred):
    y_pred, y_true = single_int64(y_pred, y_true)
    jacc = jaccard_score(y_true, y_pred, average='binary').astype('float32')
    return jacc


def _precision(y_true, y_pred):
    y_pred, y_true = single_int64(y_pred, y_true)
    p = precision_score(y_true=y_true, average='binary', y_pred=y_pred).astype('float32')
    return p


def recall_score_numpy(ytrue, ypred):
    ypred, ytrue = single_int64(ypred, ytrue)
    return recall_score(y_true=ytrue, y_pred=ypred, average='binary').astype('float32')


def _recall(y_true, y_pred):
    r = tf.numpy_function(recall_score_numpy, [y_true, y_pred], tf.float32)
    return r


def single_int64(ypred, ytrue):
    ytrue = np.where(ytrue > 0.5, 1, 0).astype(dtype=np.int64)
    ytrue = np.reshape(ytrue, [-1])
    ypred = np.where(ypred > 0.5, 1, 0).astype(dtype=np.int64)
    ypred = np.reshape(ypred, [-1])
    return ypred, ytrue


def _specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    specificity_score = true_negatives / (true_negatives + false_positives)
    return specificity_score


class MetricLayer(keras.layers.Layer):

    def __init__(self, alpha=0.25,gamma=2.0,trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.alpha = tf.constant([alpha, 1 - alpha], dtype=tf.float32)
        self.gamma = gamma

    def call(self, inputs, *args, **kwargs):
        y_true, y_pred_i = inputs
        y_pred = tf.where(y_pred_i > 0.5, 1.0, 0.0)
        self.add_metric(value=_dice(y_true, y_pred_i), name='dice')
        self.add_metric(value=_recall(y_true, y_pred), name='recall')
        self.add_metric(value=tf.numpy_function(_specificity, [y_true, y_pred], tf.float32), name='specificity')
        self.add_metric(value=tf.numpy_function(_jaccard, [y_true, y_pred], tf.float32), name='jaccard')
        self.add_metric(value=tf.numpy_function(_precision, [y_true, y_pred], tf.float32), name='precision')
        self.add_metric(value=keras.metrics.binary_accuracy(y_true, y_pred), name='accuracy')
        loss_bce = keras.losses.binary_crossentropy(y_true, y_pred_i)
        loss_bce_ = tf.reduce_mean(loss_bce)
        self.add_loss(loss_bce_)
        self.add_metric(loss_bce_, name="loss_crossentropy")

        return y_pred


class MetricsCallback(Callback):
    def __init__(self, _valid_dataset, _image_path):
        super(MetricsCallback, self).__init__()
        self.validation_data: tf.data.Dataset = _valid_dataset
        self.image_path = image_path
        self.best_dice = float('-inf')
        self.path = 'archive/Dataset_BUSI_with_GT/all/benign (1).png'


    def on_epoch_end(self, epoch, logs=None):
        dice_value = logs.get('val_dice')
        if dice_value > self.best_dice:
            self.best_dice = dice_value
            best_dice_epoch = epoch
            print("-----------------------------")
            print("best acc and epoch:", self.best_dice, best_dice_epoch)
            print("-----------------------------")

            data_list = list(iter(self.validation_data.map(lambda x, _: x)))

            pred_list = [self.model.predict(i) for i in data_list]

            index = 0
            for ori_batch, p_mask_batch in zip(data_list, pred_list):
                for o_image, o_mask, p_mask in zip(ori_batch[0], ori_batch[1], p_mask_batch):
                    total_pixels = 128 * 128
                    num_of_ones = tf.reduce_sum(o_mask)
                    ratio = num_of_ones / total_pixels
                    ratio = "{:.5f}".format(ratio)
                    fig, _ = plt.subplots(1, 3)
                    plt.subplot(1, 3, 1)
                    plt.axis('off')
                    plt.imshow(o_image)
                    plt.subplot(1, 3, 2)
                    plt.axis('off')
                    plt.imshow(o_mask, cmap='gray')
                    plt.subplot(1, 3, 3)
                    plt.axis('off')

                    p_mask_bin = np.where(p_mask > 0.5, 1.0, 0.0)
                    plt.imshow(p_mask_bin, cmap='gray')
                    fig.savefig(self.image_path + f"{index}_true{ratio}_{epoch}.png")
                    plt.close(fig)
                    index += 1

