# coding=gbk
import os
import random

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

# mpl.use('TkAgg')

img_w = 128
img_h = 128

image_sets = []
label_sets = []



def read_directory(image_directory_name, label_directory_name):
    for image_filename in os.listdir(image_directory_name):
        # image
        image_path = os.path.join(image_directory_name, image_filename)
        image_file = cv2.imread(image_path)
        image_sets.append(image_file)
    for label_filename in os.listdir(label_directory_name):
        # label
        label_path = os.path.join(label_directory_name, label_filename)
        label_file = cv2.imread(label_path)
        label_sets.append((label_file))
    return image_sets, label_sets


def elastic_transform(image, label, alpha=10, sigma=2, alpha_affine=2, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3

    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])

    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)

    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    imageA = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    xb = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    yb = map_coordinates(imageA, indices, order=1, mode='constant').reshape(shape)
    return xb, yb


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


# 均值滤波
def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


def add_noise(xb):
    sigma = random.uniform(5, 10)
    h = xb.shape[0]
    w = xb.shape[1]
    c = xb.shape[2]
    gauss = np.random.normal(0, sigma, (h, w, c))
    gauss = gauss.reshape(h, w, c)
    noisy = xb + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy


def data_augment(xb, yb):
    xb = xb * 255
    xb = xb.astype(np.uint8)
    yb = yb * 255
    yb = yb.astype(np.uint8)
    in_shape = np.shape(yb)
    # if np.random.random() < 0.80:
    #     xb = xb
    #     yb = yb
    # else:
    if np.random.random() < 0.20:
        xb, yb = rotate(xb, yb, random.uniform(0, 1) * 20)
        yb = np.reshape(yb, in_shape)
    if np.random.random() < 0.20:
        xb, yb = rotate(xb, yb, random.uniform(1, 1.05) * 340)
        yb = np.reshape(yb, in_shape)
    if np.random.random() < 0.30:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.30:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.30:
        xb = blur(xb)

    if np.random.random() < 0.25:
        xb = add_noise(xb)

    if np.random.random() < 0.35:
        xb_s = np.split(xb, 3, axis=-1)
        el_rs = list()
        for i in xb_s:
            i = np.squeeze(i)
            ib, yib = elastic_transform(i, yb)
            el_rs.append(ib)
        xb = np.stack(el_rs, axis=-1)
        yb = yib
    yb = np.reshape(yb, in_shape)
    xb = xb / 255.0
    xb = xb.astype(np.float32)
    yb = yb / 255.0
    yb = yb.astype(np.float32)
    return xb, yb
