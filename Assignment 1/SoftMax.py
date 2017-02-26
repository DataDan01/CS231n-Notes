# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:39:17 2017

@author: DA
"""

############################################
## Setup
from data_utils_py3 import load_CIFAR10
import numpy as np


X_train, y_train, X_test, y_test = load_CIFAR10('../data/cifar-10-batches-py')

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

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Centering data
mean_image = np.mean(X_train, axis = 0)

X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

# Appending static column of 1s for bias tick, making cols
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T

del mean_image, mask, num_training, num_test, num_validation

#################################################################
#### Implementation Begins Here 

# Naive inefficient version

W = np.random.randn(10, 3073) * 0.000001

def softmax_loss_naive(W, X, y, reg):
  
  scores = W.dot(X)  
  
  shift = np.max(scores) # For numerical stability 
  
  loss_num = np.exp(scores[y,np.arange(scores.shape[1])] + shift)
  
  loss_denom = np.sum(np.exp(scores) + shift, 0)
  
  data_loss = np.sum(-np.log(loss_num/loss_denom))
  
  data_loss /= scores.shape[1]
   
  return data_loss
  
# Testing to see if implementation is roughly correct
# We expect a loss close to -log(0.1) because the small random weights assign roughly equal probability to each class and there are 10 classes, the loss has the extra -log() because it makes it easier to optimize  
softmax_loss_naive(W, X_train, y_train, 0.01)  
-np.log(0.1)

# Naive version already vectorized, implementing full function with gradient

#X = X_train #X = X_train[:,0:2]
#y = y_train #y = y_train[0:2]
#reg = 0.0001

from numba import jit

@jit
def softmax_loss(W, X, y, reg):
  
  ## Computing scores and data loss ##
  scores = W.dot(X)  
  
  shift = -np.max(scores) # For numerical stability 
  
  loss_denom = np.sum(np.exp(scores + shift), 0)  
  
  all_probs = np.exp(scores + shift)/loss_denom   
  
  data_loss = all_probs[y,np.arange(all_probs.shape[1])]  
  
  data_loss = np.sum(-np.log(data_loss))  
  
  data_loss /= scores.shape[1]   
    
  ## Computing gradient over batch ##
  all_probs[y,np.arange(all_probs.shape[1])] -= 1
  
  dW = np.dot(X, all_probs.T).T/scores.shape[1]  
  
  ## Adding in regularization ##
  loss = data_loss + 0.5*reg*np.sum(W*W)  
  
  dW += reg*W  
  
  return loss, dW

# Testing function output
loss, dW = softmax_loss(W, X_train, y_train, 0.001)

# Training function on data
W = np.random.randn(10, 3073) * 0.00001

niter = 1000
learn_rate = 1e-6
reg_strength = 5e2

for i in range(niter):
    
    loss, dW = softmax_loss(W, X_train, y_train, reg_strength)
    
    print("Training loss of %.4f on iter %.0f" % (loss,i+1)) 
    
    W += -1*learn_rate*dW

pred = W.dot(X_val)

accuracy = sum(np.argmax(pred, 0) == y_val)/len(y_val) 

print("Validation accuracy of %.2f%%" % (accuracy*100)) 