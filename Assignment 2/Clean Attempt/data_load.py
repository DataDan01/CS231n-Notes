# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:27:17 2017

@author: DA
"""

exec(open('../data_utils_py3.py').read())

import numpy as np

from pathlib import Path
import os

X_train, y_train, X_test, y_test = load_CIFAR10(str(Path(os.getcwd()).parents[1])+'/data/cifar-10-batches-py')

# Subsample the data for more efficient code execution in this exercise.
num_training = 49000
num_validation = 1000
num_test = 1000

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

del mask, num_test, num_training, num_validation
