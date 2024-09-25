# coding=utf-8
import keras
from keras.layers import *
import keras.backend as K
import os

os.environ["KMP_DULLICATE_OK"] = "True"


def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat



def Channelblock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same", dilation_rate=(3, 3))(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = ReLU()(batch1)

    conv2 = Conv2D(filte, (5, 5), padding="same")(data)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = ReLU()(batch2)

    data3 = concatenate([LeakyReLU1, LeakyReLU2])
    data3 = GlobalAveragePooling2D()(data3)
    data3 = Dense(units=filte)(data3)
    data3 = BatchNormalization()(data3)
    data3 = ReLU()(data3)
    data3 = Dense(units=filte)(data3)
    data3 = Activation('sigmoid')(data3)

    a = Reshape((1, 1, filte))(data3)

    a1 = 1 - data3
    a1 = Reshape((1, 1, filte))(a1)

    y = multiply([LeakyReLU1, a])

    y1 = multiply([LeakyReLU2, a1])

    data_a_a1 = concatenate([y, y1])

    conv3 = Conv2D(filte, (1, 1), padding="same")(data_a_a1)
    batch3 = BatchNormalization()(conv3)
    LeakyReLU3 = ReLU()(batch3)
    return LeakyReLU3



def Spatialblock(data, channel_data, filte, size, Contrastive=False):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = ReLU()(batch1)

    conv2 = Conv2D(filte, (1, 1), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = ReLU()(batch2)

    data3 = add([channel_data, LeakyReLU2])
    data3 = ReLU()(data3)
    data3 = Conv2D(1, (1, 1), padding='same')(data3)
    data3 = Activation('sigmoid')(data3)

    a = expend_as(data3, filte)

    a1 = 1 - data3

    if Contrastive:
        a1 = a1**2/(a1**2+0.1)
        a1_ = expend_as(a1,filte)
        a = 1-a1
        a = expend_as(a,filte)
        y = multiply([a, channel_data])
        y1 = multiply([a1_, LeakyReLU2])

    else:
        a = a
        y = multiply([a, channel_data])


        a1 = expend_as(a1, filte)
        # 前景
        y1 = multiply([a1, LeakyReLU2])


    data_a_a1 = concatenate([y1, y])

    conv3 = Conv2D(filte, size, padding='same')(data_a_a1)
    batch3 = BatchNormalization()(conv3)
    act = ReLU()(batch3)

    return act


def HAAMLayer(data, filter2, size, Contrastive=False):
    channel_data = Channelblock(data=data, filte=filter2)

    haam_data = Spatialblock(data, channel_data, filter2, size, Contrastive=Contrastive)

    return haam_data


i2 = list()


def HAAM(data, filter1, size, Contrastive=False):
    i2.append("abc")
    # return HAAMLayer(data,filter1,size)
    return keras.models.Model(inputs=[data], outputs=[HAAMLayer(data, filter1, size, Contrastive)], name=f"HAAM_{len(i2)}")(
        data)


class AddMetric(keras.layers.Layer):
    def __init__(self, met_name):
        super(AddMetric, self).__init__()
        self.met_name = met_name

    def call(self, inputs, *args, **kwargs):
        self.add_loss(inputs)
        self.add_metric(inputs, name=self.met_name)
        return inputs
