# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:12:48 2017

@author: DA
"""

# Setup
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

# Naive random search loss function
def svm_loss_naive(W, X, y, reg):

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in range(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss
  
# generate a random SVM weight matrix of small numbers
W = np.random.randn(10, 3073) * 0.01 
loss = svm_loss_naive(W, X_train, y_train, 0.00001)
print(loss) # 8 - 9 range

# Adding in gradient
from numba import jit
 
@jit 
def svm_loss_naive(W, X, y, reg):

  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in range(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    no_wrong = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[j,:] = W[j,:] # Decreasing weights on wrong classes decreases the loss
        no_wrong += 1
        loss += margin
    dW[y[i],:] = W[y[i],:]*-1*no_wrong # Inc weights on right class dec loss

  # Adding effect of regularization to loss
  dW = dW + reg*2*W
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  return loss, dW
  
W = np.random.randn(10, 3073) * 0.0001 

loss, grad = svm_loss_naive(W, X_train, y_train, 0.00001) 

loss

### It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? Hint: the SVM loss function is not strictly speaking differentiable

# This is because the max() function isn't smooth and the derivative gets weird around the kink
# An example could be where we're right below the kink and any increase will lead to nearly a matching increase in the output (once be break zero) but the gradient marks it as zero due to the indicator function
# This is not a cause for concern and instead is a fact of life since gradients, by construction, are just tools that help us estimate things

# Instead of vectorizing, let's try Cython


%load_ext Cython

%%cython
cpdef np.ndarray[np.int32_t, ndim=2] svm_loss_cython(np.ndarray[np.int32_t, ndim=2] W, np.ndarray[np.int32_t, ndim=2] X, np.ndarray[np.int32_t, ndim=1] y, float reg):

  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in range(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    no_wrong = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[j,:] = W[j,:] # Decreasing weights on wrong classes decreases the loss
        no_wrong += 1
        loss += margin
    dW[y[i],:] = W[y[i],:]*-1*no_wrong # Inc weights on right class dec loss

  # Adding effect of regularization to loss
  dW = dW + reg*2*W
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  

  return loss, dW
