# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 19:52:56 2017


"""

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


def rgb2gray(rgb):
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

tf.reset_default_graph()
loaded = np.load(r"sprites.npy")
num_obs = loaded.shape[0]
imgs = loaded.reshape([num_obs, 120, 120, 3])
test_img = imgs[3, :, :, :].astype(np.float32)
open_cv_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
open_cv_img = cv2.resize(open_cv_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
cv2.imshow('pok2', open_cv_img)
imgs = imgs[:, 20:100, 20:100, :] #Narrow ROI to 80x80 square, should provide enough relevant information, diminishing computational cost


df = pd.read_csv(r"Pokemon.csv", index_col=0)

df = df[(df['Type 1'] == 'Fire') | (df['Type 1'] == 'Water') | (df['Type 1'] == 'Electric')]
#df = df.reset_index(drop=True)



imgs = imgs[list(df.index), :, :, :]
df = df.reset_index(drop=True)
X = imgs.reshape([-1, 80, 80, 3])
y = df['Type 1']
y = pd.get_dummies(y)

y = np.array(y)
#
#


def train():
    with tf.Graph().as_default():
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        #img_aug.add_random_rotation(max_angle=25.)
        img_aug.add_random_blur(sigma_max=3.)
        convnet = input_data(shape=[None, 80, 80, 3], name='input',
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = dropout(convnet, 0.5)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.7)
        convnet = fully_connected(convnet, 512, activation='relu')

        convnet = dropout(convnet, 0.5)

        convnet = fully_connected(convnet, len(df['Type 1'].unique()),
                                               activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=0.001,
                           loss='categorical_crossentropy')

        model = tflearn.DNN(convnet)

        model.fit(X, y, n_epoch=30,
                  validation_set=0.4, show_metric=True, run_id='Pokemon')

        model.save('pokemon.model')


#test_img = imgs[3,:,:,:]
#gray = rgb2gray(imgs)

from matplotlib import pyplot as plt
plt.imshow(test_img, interpolation='nearest')
plt.show()

