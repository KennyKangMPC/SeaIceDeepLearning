"""
Created on Mon May  4 22:59:56 2020

This is fore generating more images for training and testing

@author: Kangqi Fu
"""

from tensorflow.keras import models
from tensorflow.keras.layers import Activation, Reshape, Permute
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization

def segnet():
    img_w = 128
    img_h = 128
    n_labels = 1
    
    kernel = 3
    
    encoding_layers = [
        Convolution2D(64, kernel, padding='same', input_shape=( img_h, img_w,3)),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]

    autoencoder = models.Sequential() # segnet is actually a kind of autoencoder
    autoencoder.encoding_layers = encoding_layers
    
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
        print(l.input_shape,l.output_shape,l)

    decoding_layers = [
        UpSampling2D(),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        
        UpSampling2D(),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        
        UpSampling2D(),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        
        UpSampling2D(),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(n_labels, (1, 1), padding='valid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    autoencoder.add(Convolution2D(n_labels, (1,1), activation='sigmoid'))
    
    autoencoder.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    return autoencoder