#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 5 17:27:27 2020

This is the engine of the whole program
@author: Kangqi Fu
"""

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
from Seg_net import segnet
from U_net import unet

from results import read_images

# neural network images size
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3 #RGB

# Images path for both train and test images and their corresponding
# groundtruth labels

tr_path = './label_create/training/images/output/' # This are for augmented images
te_path = './label_create/test/images/output/' # This are for augmented images



if __name__ == '__main__':
    # list for train images, first half is labels and second half is images
    tr_file_list = sorted(os.listdir(tr_path)) 
    # list for test images, first half is labels and second half is images
    te_file_list = sorted(os.listdir(te_path)) # list for test images
        
    X_train, Y_train, X_test, Y_test = read_images(tr_file_list, te_file_list)
    
    # This is for testing the previous image preprocessing process
# =============================================================================
#     image_x = random.randint(0, int(len(tr_file_list)/2)-1)
#     imshow(X_train[image_x])
#     plt.show()
#     imshow(np.squeeze(Y_train[image_x]))
#     plt.show()
#     
#     image_x = random.randint(0, int(len(te_file_list)/2)-1)
#     imshow(X_test[image_x])
#     plt.show()
#     imshow(np.squeeze(Y_test[image_x]))
#     plt.show()
#     
# =============================================================================
    
    # Call models and then train it
    ##########################
    model = unet(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS) # Here is Unet model
    #model = segnet()
    
    #Modelcheckpoint
    model.summary()
    savePath = 'Ubestmodel.h5'

    checkpointer = ModelCheckpoint(savePath, verbose=1, save_best_only=True)
    cb = [
        EarlyStopping(patience=50, monitor='val_loss', mode='min'),
        TensorBoard(log_dir = 'logs'),
        checkpointer
        ] #save directory
    
    # model fit
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=200, callbacks=cb)
    # check details of parameters from documentations of model.fit()
    # callback use ModelCheckPoint? EarlyStopping? again, check for details for implementation
    # visualization tool, my plan is to use tensorboard which is also within the callback parameter
    
    
        
    
    
    
    
    
    