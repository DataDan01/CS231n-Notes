# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:26:34 2017

@author: DA
"""

%load_ext Cython

%%cython

import numpy as np
cimport numpy as np
import cython

def affine_relu_forward(x, np.ndarray[np.float64_t, ndim = 2] w, np.ndarray[np.float64_t, ndim = 1] b, np.ndarray[np.float64_t, ndim = 1] a):

  ### Affine component ###
  # Reshaping the input into a batch of examples as row vectors
  cdef int N = x.shape[0]
  cdef int D = x.size//N

  cdef np.ndarray[np.float64_t, ndim = 2] x_flat = np.reshape(x, (N, D))
  
  # Computing dot product
  cdef np.ndarray[np.float64_t, ndim = 2] aff_out = np.dot(x_flat, w) + b  

  ### Parametric ReLU Component ###
  cdef np.ndarray[np.float64_t, ndim = 2] out = np.maximum(a*aff_out,aff_out) 

  return out

# affine_relu_forward(x = X_train[0:5], w = np.random.randn(3072,100), b = np.random.randn(100), a = np.random.randn(100))

#

def affine_relu_backward(np.ndarray[np.float64_t, ndim = 2] dout, x, np.ndarray[np.float64_t, ndim = 2] w, np.ndarray[np.float64_t, ndim = 1] b, np.ndarray[np.float64_t, ndim = 1] a):

  ### Affine Component ###
  cdef int N = x.shape[0]
  cdef int D = x.size//N   

                                            #N*D  N*M     M*D
  cdef np.ndarray[np.float64_t, ndim = 2] dx_aff = dout.dot(w.T)
  
                                          #D*M   D*N        N*M
  cdef np.ndarray[np.float64_t, ndim = 2] dw = x.T.dot(dout)
  
                                          #M*1  M*N           N*1
  cdef np.ndarray[np.float64_t, ndim = 1] db = dout.T.dot(np.ones(N))
  
  ### Parametric ReLU Component ###  
  cdef np.ndarray[np.float64_t, ndim = 2] da = x[x <= 0] ###???
  
  cdef np.ndarray[np.float64_t, ndim = 2] out = 1. * (x > 0)
  
  # Reshaping a and adding contents where we are below the kink
  cdef np.ndarray[np.float64_t, ndim = 2] rep_a = np.array([a]*out.shape[1])
  
  out[out == 0] += rep_a[out == 0]  
  
  cdef np.ndarray[np.float64_t, ndim = 2] dx = out * dout ###  

  return dx, dw, db, da
  
# affine_relu_backward(dout = dloss, x = cache['cache_mid_x'][:,:,3], w = all_params['W5'], b = all_params['b5'], a = all_params['a5'])  
  
# 

def softmax_loss(np.ndarray[np.float64_t, ndim = 2] x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  cdef np.ndarray[np.float64_t, ndim = 2] probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  
  cdef int N = x.shape[0]
  
  cdef np.float64_t loss = -np.sum(np.log(probs[np.arange(N), y]))/N
  
  cdef np.ndarray[np.float64_t, ndim = 2] dx = probs.copy()
  dx[np.arange(N), y] -= 1.0
  dx /= N
  
  return loss, dx