# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:09:29 2017

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

# initialize parameters randomly
h = 100 # size of hidden layer
W = 0.00001 * np.random.randn(h,X_train.shape[0])
W2 = 0.00001 * np.random.randn(h,10)

hidden_layer = np.maximum(0, np.dot(W, X_train))
scores = np.dot(W2.T, hidden_layer)

#X = X_train #X = X_train[:,0:2]
#y = y_train #y = y_train[0:2]
#reg = 0.0001

from numba import jit

@jit
def two_layer_nn(W, W2, X, y, reg):
  
  ## Computing scores and total loss ##
  hidden_layer = np.maximum(0, np.dot(W, X))

  scores = np.dot(W2.T, hidden_layer)  
  
  shift = -np.max(scores) # For numerical stability 
  
  loss_denom = np.sum(np.exp(scores + shift), 0)  
  
  all_probs = np.exp(scores + shift)/loss_denom   
  
  data_loss = all_probs[y,np.arange(all_probs.shape[1])]  
  
  data_loss = np.sum(-np.log(data_loss))  
  
  data_loss /= scores.shape[1]    
  
  loss = data_loss + 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)   
  
  ## Computing gradient over batch ## 
  # Last layer first
  all_probs[y,np.arange(all_probs.shape[1])] -= 1
  
  all_probs /= scores.shape[1]  
  
  dW2 = np.dot(hidden_layer, all_probs.T)    
  
  dW2 += reg*W2
  
  ## Backprop into first layer ##
  
  # Backprop from second deriv into second weights
  dhidden = np.dot(all_probs.T, W2.T)  
  
  # Set derivative to first layer
  dhidden[hidden_layer.T <= 0] = 0    
  
  # Backprop into first layer
  dW = np.dot(X, dhidden).T    
  
  dW += reg*W  
  
  return loss, dW, dW2
  
loss, dW, dW2 = two_layer_nn(W, W2, X_train, y_train, 0.001)

# Training function on data
h = 512 # size of hidden layer
W = 0.0001 * np.random.randn(h,X_train.shape[0])
W2 = 0.0001 * np.random.randn(h,10)

niter = 10000
batch_size = 128
learn_rate = 5e-4
reg_strength = 1e-4

# About 50% accuracy
for i in range(niter):
    
    inx = np.random.randint(0,len(y_train),batch_size)
    
    loss, dW, dW2 = two_layer_nn(W, W2, X_train[:,inx], y_train[inx], reg_strength)
    
    print("Training loss of %.4f on iter %.0f" % (loss,i+1)) 
    
    W += -1*learn_rate*dW
    W2 += -1*learn_rate*dW2

hidden_layer = np.maximum(0, np.dot(W, X_val))
pred = np.dot(W2.T, hidden_layer)

accuracy = sum(np.argmax(pred, 0) == y_val)/len(y_val) 

print("Validation accuracy of %.2f%%" % (accuracy*100))   