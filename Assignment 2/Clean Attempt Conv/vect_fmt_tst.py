# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:03:41 2017

@author: Stat-Comp-01
"""

import os
import numpy as np

os.chdir('C:/Users/Stat-Comp-01/OneDrive/Python/CS231n/Assignment 2/Clean Attempt Conv')

# Getting the data
exec(open('./data_load.py').read())

x = X_train[0:10,]
w = np.random.randn(5,3,3,3)

npad = (w.shape[3]-1)//2

x_padded = np.pad(x, 
                  npad, 
                  mode = 'constant', 
                  constant_values = 0)

x_padded = np.delete(x_padded, np.array([0,x_padded.shape[0]-1]), axis = 0)
x_padded = np.delete(x_padded, np.array([0,x_padded.shape[3]-1]), axis = 3)


# Testing
x_skinny_raw = np.reshape(X_train[0],np.prod(X_train[0].shape))
x_skinny_padded = np.reshape(x_padded[0],np.prod(x_padded[0].shape))

x_fat_raw = X_train[0]
x_fat_padded = x_padded[0]

# Need a systematic way to convert each image to a zero padded column
# Then create index for filter to pass over