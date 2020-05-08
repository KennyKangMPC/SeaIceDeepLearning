#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:59:56 2020

This is fore generating more images for training and testing

@author: Kangqi Fu
"""

import Augmentor as ag

def augmenting(im, la):
    """
    This method is to add augmentation pipeline
    """
    p = ag.Pipeline(im)
    p.ground_truth(la)
    p.rotate(probability=0.6, max_left_rotation=25, max_right_rotation=25)
    p.skew(probability=0.5)
    p.random_distortion(probability=0.5, grid_width=10, grid_height=10, magnitude=8)
    p.shear(probability=0.5, max_shear_left=20, max_shear_right=20)
    p.flip_random(probability=0.4)
    return p
    
if __name__ == '__main__':
    # for two parts, training and testing
    # training
    impath = "./label_create/training/images"  # image path
    glpath = "./label_create/training/labels"  # ground true lable path
    n = 1000 # number of samples
    ptr = augmenting(impath,glpath)
    ptr.sample(n) # use GPU to boost speed
    
    # testing
    impath = "./label_create/test/images"  # image path
    glpath = "./label_create/test/labels"  # ground true lable path
    nt = 200 # number of samples
    pte = augmenting(impath,glpath)
    pte.sample(nt) # use GPU to boost speed