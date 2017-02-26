# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:12:48 2017

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

#################################################################
#### Implementation Begins Here (incorrect)

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
loss = svm_loss_naive(W, X_train, y_train, 0.1)
print(loss) # 8 - 9 range

# Adding in gradient
from numba import jit
 
@jit 
def svm_loss_naive(W, X, y, reg, grad = True):

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
        dW[j,:] += W[j,:] # Decreasing weights on wrong classes decreases the loss
        no_wrong += 1
        loss += margin
    dW[y[i],:] += W[y[i],:]*-1*no_wrong # Inc weights on right class dec loss

  # Adding effect of regularization to loss
  dW = dW + reg*2*W
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)


  if grad == True:
      return loss, dW
      
  if grad == False:
      return loss
  
W = np.random.randn(10, 3073) * 0.001 

loss, grad = svm_loss_naive(W, X_train, y_train, 5e4) 

loss

### It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? Hint: the SVM loss function is not strictly speaking differentiable

# This is because the max() function isn't smooth and the derivative gets weird around the kink
# An example could be where we're right below the kink and any increase will lead to nearly a matching increase in the output (once be break zero) but the gradient marks it as zero due to the indicator function
# This is not a cause for concern and instead is a fact of life since gradients, by construction, are just tools that help us estimate things

# Vectorizing 
X = X_train
y = y_train
reg = 0.1

@jit
def svm_loss_vectorized(W, X, y, reg, grad = True):

  # Compute the scores
  scores = W.dot(X)
 
  # Extracting correct scores
  correct_scores = scores[y,np.arange(scores.shape[1])]
 
  # Subtract scores of correct class and add margin
  margins = scores - correct_scores + 1

  # Remove margins over correct class and negative margins
  margins[np.logical_or(margins == 1, margins <= 0)] = 0
  
  # Compute total loss, scale down by number of classes
  loss = np.sum(margins)/margins.shape[1]

  if grad == True:
      # Add in regularization loss only for training
      loss += reg*np.sum(W * W)
      # initialize the gradient as zero and setup
      dW = np.zeros(W.shape) 
      margins[margins > 0] = 1 
    
      # Difficult to vectorize this part due to modification in place
      sum_wrong = np.sum(margins, 0)
    
      # Accumulate gradient for all incorrect classes
      dW += margins.dot(X.T)

      for i in range(scores.shape[1]):
          dW[y[i],:] += -1*X[:,y[i]]*sum_wrong[i]
    
      # Adding regularization impact to gradient too
      dW += reg*2*W

      dW /= margins.shape[1]

  if grad == True:
      return loss, dW
      
  
  if grad == False:
      return loss

# Testing
W = np.random.randn(10, 3073) * 0.000001 

%%timeit
loss1, grad1 = svm_loss_naive(W, X_train, y_train, 0.1) 

%%timeit
loss2, grad2 = svm_loss_vectorized(W, X_train, y_train, 0.1)

# Very close to zero
loss1-loss2

np.sum(grad2) - np.sum(grad1)

np.linalg.norm(grad1 - grad2, ord = "fro")

## Implementing Prediction Algorithm
# Initializing random weight matrix
W = np.random.randn(10, 3073) * 0.00001 

@jit
def svm_train(W, X, y, learning_rate, reg, num_iters, verbose = True):
    
    for i in range(num_iters):
        
        loss, grad = svm_loss_vectorized(W, X, y, reg)
        
        if(verbose == True):
            print("Training loss of %.4f on iter %.d" % (loss,i+1))
        
        W += -1*learning_rate*grad

    val_loss = svm_loss_vectorized(W, X_val, y_val, reg, grad = False)    

    print("Validation data loss of %.10f" % (val_loss))    
    
    pred = W.dot(X_val)

    accuracy = sum(np.argmax(pred, 0) == y_val)/len(y_val)  
    
    print("Validation accuracy of %.2f percent" % (accuracy*100))    
    
    return W
    
W = svm_train(W, X_train, y_train, 5e-5, 1e5, 100)

##########################################################################
# Correct implementation below

# Maybe try svm function for one example at a time and then apply to array
from numba import jit

@jit
def svm_loss_indiv(W, X, y, reg):
    
    # Initialization
    dW = np.zeros(W.shape)
    loss = 0    
    
    # Computing scores and clipping values
    scores = W.dot(X)
    
    margins = scores - scores[y] + 1
    
    margins[np.logical_or(margins == 1, margins <= 0)] = 0 
    
    loss += sum(margins)
    
    # Preparing for gradient calcs    
    margins[margins > 0] = 1    
    
    margins = margins.astype(int)      
    
    # Accumulating gradient for the correct class
    dW[y,:] += -1*sum(margins)*X
    
    # Accumulating gradient for incorrect classes
    dW[np.arange(dW.shape[0])!=margins] += X
    
    # Adding regularization gradient
    dW += abs(2*reg*W)    
    
    return loss, dW    

# Testing various hyper-parameters
W = np.random.randn(10, 3073) * 1e-8

import random as rn

learning_rate = 1e-11
reg = 1e2
niter = 10000

tot_loss = 0

for i in range(niter):

    inx = rn.choice(range(X_train.shape[1]))    
    
    loss, grad = svm_loss_indiv(W, X_train[:,inx], y_train[inx], 5e4)
    
    tot_loss += loss    
    
    W += -1*learning_rate*grad
    
pred = W.dot(X_val)

accuracy = sum(np.argmax(pred, 0) == y_val)/len(y_val)  

print("Average data loss of %.4f" % (tot_loss/niter))
print("Validation accuracy of %.2f percent" % (accuracy*100))  
    
