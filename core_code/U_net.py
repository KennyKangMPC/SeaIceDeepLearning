#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 1 10:24:12 2020

@author: Kangqi Fu
"""
import tensorflow as tf # can use sequential format as well (pytorch is another package)

# U-net architecture (Put it to other class)
def unet(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, nClasses=1):
    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)) #input value has to be float point value 
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # convert type

    #Contraction path
    c1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(64,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    
    c2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(128,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.1)(c3)
    c3 = tf.keras.layers.Conv2D(256,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    
    c4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(512,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    
    c5 = tf.keras.layers.Conv2D(1024,(3,3), activation='relu',kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(1024,(3,3), activation='relu',kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansion path
    u6 = tf.keras.layers.Conv2DTranspose(512,(2,2),strides=(2,2),padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6,c4])
    c6 = tf.keras.layers.Conv2D(512,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(512,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7,c3])
    c7 = tf.keras.layers.Conv2D(256,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(256,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    u8 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8,c2])
    c8 = tf.keras.layers.Conv2D(128,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(128,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9,c1])
    c9 = tf.keras.layers.Conv2D(64,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(64,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    #ouput layer
    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)
    
    model = tf.keras.Model(inputs=[inputs],outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #binary classification?
    return model

# To run tensorboard, I check the documentation. After running the code, use:
#!tensorboard --logdir=logs/ --host localhost --port 8088
