#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:48:57 2020

@author: Kangqi Fu
"""
from tensorflow.keras.models import load_model
import numpy as np
import os
from tqdm import tqdm # a progress bar while doing for loop: visualize the process
from eval_matric import pixel_accuracy, mean_accuracy, mean_IOU, frequency_weighted_IOU
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

# neural network images size
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3 #RGB

# Images path for both train and test images and their corresponding
# groundtruth labels

tr_path = './label_create/training/images/output/' # This are for augmented images
te_path = './label_create/test/images/output/' # This are for augmented images

def read_images(tr, te):
    """
    X is images
    Y is ground truth labels
    """
    
    len_tr = int(len(tr)/2)
    len_te = int(len(te)/2)
    X_tr = np.zeros((len_tr, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    Y_tr = np.zeros((len_tr, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    
    X_te = np.zeros((len_te, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    Y_te = np.zeros((len_te, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    
    # read train images
    for i in tqdm(np.arange(len_tr)):
        Xpath = tr_path + tr[i+len_tr]
        Ypath = tr_path + tr[i]
        img = imread(Xpath)[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
        X_tr[i] = img
        label = imread(Ypath)[:, :, :IMG_CHANNELS]
        label = resize(label, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
        Y_tr[i] = label
    
    # read test images
    for i in tqdm(np.arange(len_te)):
        Xpath = te_path + te[i+len_te]
        Ypath = te_path + te[i]
        img = imread(Xpath)[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
        X_te[i] = img
        label = imread(Ypath)[:, :, :IMG_CHANNELS]
        label = resize(label, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
        Y_te[i] = label
    print('Image preparing Finish')
    return X_tr, Y_tr, X_te, Y_te

if __name__ == '__main__':
    # list for train images, first half is labels and second half is images
    tr_file_list = sorted(os.listdir(tr_path)) 
    # list for test images, first half is labels and second half is images
    te_file_list = sorted(os.listdir(te_path)) # list for test images
        
    X_train, Y_train, X_test, Y_test = read_images(tr_file_list, te_file_list)
    
    #Evaluate models using categorical_crossentropy accuracy
    
    ##########################
    # Test models
    
    # load models
    best_models = load_model('Ubestmodel.h5')
    test_loss = best_models.evaluate(X_test, Y_test)[0]
    test_acc = best_models.evaluate(X_test, Y_test)[1]
    print('Test loss: %.3f' % (test_loss))
    print('Test accuracy: %.3f' % (test_acc))
    
    
    
    preds_train_t = best_models.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val_t = best_models.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    
    # Here need to rediclare four images
    # Here for the test preds, I only will use the four original images so here
    # redeclare the path and save them to the numpy arrays and train them
    impath = "./label_create/test/images/"  # test image path
    glpath = "./label_create/test/labels/"  # test ground true lable path
    ims = sorted(os.listdir(impath)) 
    gls = sorted(os.listdir(glpath))
    
    X_rte = np.zeros((len(gls), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_rte = np.zeros((len(gls), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    
    for i in tqdm(np.arange(len(gls))):
        Xpath = impath + ims[i+1]
        Ypath = glpath + gls[i]
        img = imread(Xpath)[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
        X_rte[i] = img
        label = imread(Ypath)[:, :, :IMG_CHANNELS]
        label = resize(label, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range = True)
        Y_rte[i] = label
    
    
    preds_test_t = best_models.predict(X_rte, verbose=1) # Only need to use those four images not augmented images

    # evaluate results using four methods provided by paper: (Only do test set)
    
    # declare four matrices to store the feedback of each graphs
#    pas = np.zeros(len(Y_rte))
#    mas = np.zeros(len(Y_rte))
    mious = np.zeros(len(Y_rte))
#    fwious = np.zeros(len(Y_rte))
    
# =============================================================================
#     for i in np.arange(len(Y_rte)):
#         pas[i] = pixel_accuracy(preds_test_t[i], Y_rte[i])
#         mas[i] = mean_accuracy(preds_test_t[i], Y_rte[i])
#         mious[i] = mean_IOU(preds_test_t[i], Y_rte[i])
#         fwious[i] = frequency_weighted_IOU(preds_test_t[i], Y_rte[i])
# =============================================================================
    
    # display the results and then printout the corresponding ori and gd images
    for i in np.arange(len(Y_rte)):
# =============================================================================
#         print('Image: ', i)
#         print('Pixel accuracy: %.3f' % (pas[i]))
#         print('Mean accuracy: %.3f' % (mas[i]))
#         print('Mean IOU: %.3f' % (mious[i]))
#         print('Frequency Weighted IOU: %.3f' % (fwious[i]))
# =============================================================================
        imshow(X_rte[i]) # original image
        imshow(Y_rte[i]) # ground truth images
        imshow(preds_test_t[i]) # nn trained result
        plt.show()