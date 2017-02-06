# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# This is my own attempt at the kNN problem from the class
# This script is meant to be self contained, all answers and work will be here

# Setup
import random
import numpy as np
from data_utils_py3 import load_CIFAR10
import matplotlib.pyplot as plt

# Loading in CIFAR-10 data
X_train, y_train, X_test, y_test = load_CIFAR10('../data/cifar-10-batches-py')

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Show a few examples from the training set
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Subsample the data for more efficient code execution in this exercise
num_training = 10000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 1000
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Going to keep kNN code outside object for easier use
# First we must compute the distances between all test examples and all train examples

# Nte x Ntr matrix where each element (i,j) is the distance between the i-th test and j-th train example
import scipy.spatial.distance as dist
import statistics as st

dist_mat = dist.cdist(X_test, X_train, 'minkowski', 1)

# Test * Train
print(dist_mat.shape)

plt.imshow(dist_mat, interpolation = 'none')

# What is the cause behind the distinctly visible rows?

# Rows are distinctly visible when a test example is either relatively far (red) or close (blue) to most of the training examples

# What causes the columns?

# This is when training examples are relatively far or close to test examples, we have more interest in the former than the latter since we want to predict on new data

# Find lowest k matches by distance, remember their indices
# Find all of their labels
# Majority vote

def kNN_pred(dist_mat, pred_inx, k):
    test_inx = dist_mat[pred_inx,].argsort()[:k]
    try:
        pred = st.mode(y_train[test_inx])
    except:
        pred = y_train[test_inx][0]
    return pred
    
kNN_pred(dist_mat, pred_inx = 23, k = 100)

# Test predictor on data set, look at accuracy
preds = np.zeros(X_test.shape[0])

for ind in range(num_test):
    preds[ind] = kNN_pred(dist_mat, pred_inx = ind, k = 7)

accuracy = np.sum(preds == y_test)/num_test

print("Got accuracy of %f" % (accuracy))

##################################################
# Setting up a self contained generic classifier #
##################################################
import numpy as np
import scipy.spatial.distance as dist
import statistics as st
from data_utils_py3 import load_CIFAR10

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
      pass

  def train(self, X, y):
      # These should already be flattened
      self.X_train = X
      self.y_train = y

  def predict(self, k, X_test, y_test):
      
      self.dist_mat = dist.cdist(X_test, self.X_train, 'minkowski', 1)
         
      def indiv_pred(pred_inx, k):
        test_inx = self.dist_mat[pred_inx,].argsort()[:k]
        try:
            pred = st.mode(self.y_train[test_inx])
        except:
            pred = self.y_train[test_inx][0]
        return pred
        
      preds = np.zeros(X_test.shape[0])
    
      for ind in range(X_test.shape[0]):
            preds[ind] = indiv_pred(pred_inx = ind, k = k)
            
      self.preds = preds
      self.accuracy = np.sum(preds == y_test)/X_test.shape[0]
      
      #return [self.accuracy, self.preds]
      return self.accuracy
    
# Loading in data
X_train, y_train, X_test, y_test = load_CIFAR10('../data/cifar-10-batches-py')

# Reshaping and cutting down to size
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Initializing and training classifier, testing out before CV
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

classifier.predict(3, X_test, y_test)

####################
# Cross Validation #
####################
num_folds = 5
k_choices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

acc_mat = np.zeros([num_folds-1, len(k_choices)])

import random as rn

inx_rnd = list(range(X_train.shape[0]))

rn.shuffle(inx_rnd)

folds_inx = np.array_split(inx_rnd, num_folds)

###
classifier = KNearestNeighbor()

classifier.train(X_train[folds_inx[0],:], y_train[folds_inx[0]])

for k in range(acc_mat.shape[1]):

    for i in range(acc_mat.shape[0]):
    
       acc_mat[i,k] = classifier.predict(k_choices[k], X_train[folds_inx[i+1],:], y_train[folds_inx[i+1]])
      
# Let's take a look at the performance
np.mean(acc_mat, axis = 0) 

import matplotlib.pyplot as plt

for i in range(acc_mat.shape[1]):
    plt.scatter([k_choices[i]]*4, acc_mat[:,i])
    
plt.plot(k_choices, np.mean(acc_mat, axis = 0))
plt.title('kNN Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('k Value')
plt.xticks(k_choices)

# Looks like 7 is the best value for k, let's test it out
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# 30%, three times better than random guessing
classifier.predict(7, X_test, y_test) 