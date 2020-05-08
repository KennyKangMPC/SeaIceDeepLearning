#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 3 04:34:03 2020

@author: Kangqi Fu
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer, Activation, Reshape
from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D

def segnet(nClasses=3, IMG_WIDTH = 128, IMG_HEIGHT = 128, IMG_CHANNELS = 3, label_W=128, label_H=128):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    
    model = Sequential()
    model.add(Layer(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)))
    
    # Encoder
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Decoder
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(512, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(128, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='same'))
    model.add(BatchNormalization())
    
    model.add(Convolution2D(nClasses, 1, 1, border_mode='same',))
    model.add(Reshape((label_W * label_H,)))
    model.add(Activation('softmax'))
    model.add(Reshape((label_W, label_H, nClasses)))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Here may change
    return model
