# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:13:36 2017

@author: Weidi Xie, Dominic Waithe

@description: 
This is the file to create the model, similar as the paper, but with batch normalization and skip layers,
make it more easier to train.
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    Merge,
    merge,
    Dropout,
    Reshape,
    Permute,
    Dense,
    UpSampling2D,
    Lambda,
    Flatten,
    
    )
from keras import backend as K

from keras.optimizers import SGD, RMSprop
from keras.layers.convolutional import (
    Convolution2D)
import tensorflow as tf
from keras.layers.pooling import (
    MaxPooling2D,
    AveragePooling2D
    )
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

weight_decay = 1e-4

def _conv_bn_relu(nb_filter, row, col, subsample = (1,1)):
    def f(input):
        conv_a = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',bias = False,
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(input)
        norm_a = BatchNormalization()(conv_a)
        act_a = Activation(activation = 'relu')(norm_a)
        conv_b = Convolution2D(nb_filter, row, col, subsample = subsample,
                               init = 'orthogonal', border_mode = 'same',bias = False,
                               W_regularizer = l2(weight_decay),
                               b_regularizer = l2(weight_decay))(act_a)
        norm_b = BatchNormalization()(conv_b)
        act_b = Activation(activation = 'relu')(norm_b)
        return act_b
    return f

def net_base(input, nb_filter = 64):
    # Stream
    block1 = _conv_bn_relu(nb_filter,3,3)(input)
    pool1 = MaxPooling2D(pool_size=(2,2))(block1)
    # =========================================================================
    block2 = _conv_bn_relu(nb_filter,3,3)(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
    # =========================================================================
    block3 = _conv_bn_relu(nb_filter,3,3)(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
    # =========================================================================
    block4 = _conv_bn_relu(nb_filter,3,3)(pool3)
    up4 = merge([UpSampling2D(size=(2, 2))(block4), block3], mode='concat', concat_axis=-1)
    # =========================================================================
    block5 = _conv_bn_relu(nb_filter,3,3)(up4)
    up5 = merge([UpSampling2D(size=(2, 2))(block5), block2], mode='concat', concat_axis=-1)
    # =========================================================================
    block6 = _conv_bn_relu(nb_filter,3,3)(up5)
    up6 = merge([UpSampling2D(size=(2, 2))(block6), block1], mode='concat', concat_axis=-1)
    # =========================================================================
    block7 = _conv_bn_relu(nb_filter,3,3)(up6)
    return block7

def buildModel (input_dim):
    # This network is used to pre-train the optical flow.
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = net_base (input_, nb_filter = 64 )
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, bias = False, activation='linear',\
                                  init='orthogonal',name='pred',border_mode='same')(act_)
    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model


def buildModel_fft (input_dim):
    # This network is used to pre-train the optical flow.
    input_ = Input (shape = (input_dim))
    # =========================================================================
    act_ = net_base (input_, nb_filter = 64 )
    # =========================================================================
    density_pred =  Convolution2D(1, 1, 1, bias = False, activation='linear',\
                                  init='orthogonal',name='pred',border_mode='same')(act_)

    imageMean = tf.reduce_mean(density_pred)
    node4 = tf.reshape(density_pred,[1,1,128,128])
    fftstack = tf.fft2d(tf.complex(node4,tf.zeros((1,1,128,128))))
    out = (tf.cast(tf.complex_abs(tf.ifft2d(fftstack*tf.conj(fftstack))), dtype=tf.float32)/imageMean**2/(128*128))-1
   

    def count(out):
        sigma = 4.0
        rough_sig = sigma*2.3588*0.8493218
        return (128*128)/((tf.reduce_max(out)-tf.reduce_min(out))*np.pi*(rough_sig**2))
    

    
    #lam.build((1,1))
    y_true_cast = K.placeholder(shape=(1,1,128,128), dtype='float32')
    #K.set_value(y_true_cast,out)

    # =========================================================================
    model = Model (input = input_, output = density_pred)
    opt = SGD(lr = 1e-2, momentum = 0.9, nesterov = True)
    model.compile(optimizer = opt, loss = 'mse')
    return model
