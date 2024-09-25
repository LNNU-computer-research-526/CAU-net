import math

import keras.layers
from keras.layers import Conv2DTranspose as Deconv2D
from HAAM import *
from my_metric import *

import tensorflow as tf



def max_pool(input):
    return keras.layers.MaxPooling2D()(input)


def up_sampling(input, filter):
    return Deconv2D(filter, (3, 3), strides=(2, 2), padding='same')(input)



class PatchBlock(keras.layers.Layer):
    def __init__(self):
        super(PatchBlock, self).__init__()
        self.channels = None
        self.patch_height = None
        self.patch_width = None

    def build(self, input_shape):
        self.patch_width = input_shape[1] // 2
        self.patch_height = input_shape[2] // 2
        self.channels = input_shape[3]

    def call(self, inputs, *args, **kwargs):
        image_patches = tf.image.extract_patches(
            images=inputs,
            sizes=(1, self.patch_width, self.patch_height, 1),
            strides=(1, self.patch_width, self.patch_height, 1),
            rates=(1, 1, 1, 1),
            padding="VALID"
        )
        reshape_patches = tf.reshape(image_patches, [-1, self.patch_width, self.patch_height, self.channels])
        return reshape_patches



class RePatchBlock(keras.layers.Layer):
    def __init__(self):
        super(RePatchBlock, self).__init__()
        self.channels = None
        self.patch_height = None
        self.patch_width = None

    def build(self, input_shape):
        self.patch_width = input_shape[1]
        self.patch_height = input_shape[2]
        self.channels = input_shape[3]

    def call(self, inputs, *args, **kwargs):
        patches_image = tf.reshape(inputs, [-1, 2, 2, self.patch_width, self.patch_height, self.channels])

        un_stack_patches = tf.unstack(patches_image, axis=1)
        ab_widths = tf.unstack(un_stack_patches[0], axis=2)
        cd_widths = tf.unstack(un_stack_patches[1], axis=2)
        patch_widths = tf.stack(ab_widths + cd_widths, axis=2)

        un_stack_patches = tf.unstack(patch_widths, axis=1)
        ac_heights = tf.unstack(un_stack_patches[0], axis=2)
        bd_heights = tf.unstack(un_stack_patches[1], axis=2)
        patch_heights = tf.stack(ac_heights + bd_heights, axis=2)

        return patch_heights


def linear_functions(input, filter):
    width = int(math.sqrt(input.get_shape()[-2]))
    input = keras.layers.Reshape(target_shape=[width, width, -1])(input)
    conv = keras.layers.Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding="same")(input)
    l2norm = keras.layers.BatchNormalization()(conv)
    sum_res = keras.layers.Add()([conv, l2norm])
    return keras.layers.Reshape(target_shape=[width ** 2, -1])(sum_res)


class TSAddLoss(keras.layers.Layer):

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.gam = 3e-2

    def call(self, inputs, *args, **kwargs):
        fs_, ft = inputs
        loss_TAT = tf.norm(fs_ - ft, ord=2, axis=1)
        loss_TAT = tf.reduce_mean(loss_TAT, axis=[1, 2])
        loss_TAT = self.gam * loss_TAT
        self.add_loss(loss_TAT)
        self.add_metric(loss_TAT, name="loss_TAT")
        return inputs


def student_feature_by_teacher(teacher, student):
    width = teacher.get_shape()[-2]
    out_channel = student.get_shape()[-1]
    student = keras.layers.Reshape(target_shape=[width ** 2, -1])(student)
    teacher = keras.layers.Reshape(target_shape=[width ** 2, -1])(teacher)
    gam_student = linear_functions(student, out_channel)
    sita_teacher = linear_functions(teacher, out_channel)
    fai_student = linear_functions(student, out_channel)
    mul_factor = keras.layers.Multiply()([gam_student, sita_teacher])
    sum_factor = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1))(mul_factor)
    softmax_out = keras.layers.Softmax()(sum_factor)
    softmax_out_reshape = keras.layers.Reshape(target_shape=[-1, 1])(softmax_out)
    mul_result = keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([softmax_out_reshape, fai_student])
    return keras.layers.Reshape(target_shape=[width, width, -1])(mul_result)



def TsBlock(teacher, student):
    patch_t = PatchBlock()(teacher)
    patch_s = PatchBlock()(student)
    student_ = student_feature_by_teacher(patch_t, patch_s)
    repatch_s = RePatchBlock()(student_)
    concatenate_st = concatenate([repatch_s, teacher], axis=3)
    return concatenate_st


def AAUnet():

    image_input = keras.layers.Input((image_size, image_size, image_channel))
    mask_input = keras.layers.Input((image_size, image_size, 1))
    k_count_input = keras.layers.Input(1)


    skip_teachers = list()
    feature_down = image_input

    for i in range(4):
        feature_down = HAAM(feature_down, 32 * 2 ** i, (3, 3))
        feature_down = HAAM(feature_down, 32 * 2 ** i, (3, 3))
        skip_teachers.append(feature_down)
        feature_down = MaxPooling2D((2, 2))(feature_down)


    feature_down = HAAM(feature_down, 32 * 2 ** 4, (3, 3))
    feature_up = HAAM(feature_down, 32 * 2 ** 4, (3, 3))
    feature_up = HAAM(feature_up, 256, (3, 3))
    #
    feature_up = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(feature_up)
    feature_up = concatenate([feature_up, skip_teachers[3]], axis=3)
    feature_up = HAAM(feature_up, 128, (3, 3))
    feature_up = HAAM(feature_up, 128, (3, 3))

    for i in reversed(range(1, 3)):
        feature_up = Conv2DTranspose(32 * 2 ** i, (2, 2), strides=(2, 2), padding='same')(feature_up)
        feature_up = TsBlock(teacher=skip_teachers[i], student=feature_up)
        feature_up = HAAM(feature_up, 32 * 2 ** (i - 1), (3, 3))
        feature_up = HAAM(feature_up, 32 * 2 ** (i - 1), (3, 3))

    feature_up = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(feature_up)
    feature_up = HAAM(feature_up, 32, (3, 3),Contrastive=True)


    feature_up = Conv2D(1, (1, 1), padding="same", activation='sigmoid', name='sigmoid')(feature_up)


    conv2 = MetricLayer()([mask_input, feature_up])
    return keras.Model(inputs=[image_input, mask_input, k_count_input], outputs=[conv2])





