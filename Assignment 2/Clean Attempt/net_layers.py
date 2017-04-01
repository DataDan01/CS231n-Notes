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

def affine_forward_a(np.ndarray[np.float64_t, ndim = 4] x, np.ndarray[np.float64_t, ndim = 2] w, np.ndarray[np.float64_t, ndim = 1] b):
  
  # Reshaping the input into a batch of examples as row vectors
  cdef int N = x.shape[0]
  cdef int D = x.size//N

  cdef np.ndarray[np.float64_t, ndim = 2] x_flat = np.reshape(x, (N, D))
  
  # Computing dot product
  cdef np.ndarray[np.float64_t, ndim = 2] out = np.dot(x_flat, w) + b

  cache = x, w, b
  
  return out, cache

def affine_forward_b(np.ndarray[np.float64_t, ndim = 2] x, np.ndarray[np.float64_t, ndim = 2] w, np.ndarray[np.float64_t, ndim = 1] b):
  
  # Reshaping the input into a batch of examples as row vectors
  cdef int N = x.shape[0]
  cdef int D = x.size//N

  cdef np.ndarray[np.float64_t, ndim = 2] x_flat = np.reshape(x, (N, D))
  
  # Computing dot product
  cdef np.ndarray[np.float64_t, ndim = 2] out = np.dot(x_flat, w) + b

  cache = x, w, b
  
  return out, cache

def affine_forward(x, np.ndarray[np.float64_t, ndim = 2] w, np.ndarray[np.float64_t, ndim = 1] b):
    """
  Computes the forward pass for an affine (fully-connected) layer.
  
  This strings together two cases in cython because the dimension
  of the np array cannot be dynamic.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
    if len(x.shape) == 4:
        return affine_forward_a(x,w,b)
    if len(x.shape) == 2:
        return affine_forward_b(x,w,b)
  
# affine_forward(X_train[0:10],np.random.randn(3072,100),np.random.randn(100))
# affine_forward(np.random.randn(10,126),np.random.randn(126,100),np.random.randn(100))

#%%cython

#import numpy as np
#cimport numpy as np
#import cython
  
def affine_backward_a(np.ndarray[np.float64_t, ndim = 2] dout, tuple cache):

  x, w, b = cache
  
  # Saving down dimensionality and reshaping
  cdef int N = x.shape[0]
  cdef int D = x.size//N 
 
  cdef np.ndarray[np.float64_t, ndim = 2] x_flat = np.reshape(x, (N, D))
  
  # Propogating derivative through skinny vector format and spreading out
  #N*D  N*M     M*D
  cdef np.ndarray[np.float64_t, ndim = 2] dx_s = dout.dot(w.T)
  
  # Accounting for derivatives to all intermediate layers
  cdef np.ndarray[np.float64_t, ndim = 2] dx = np.reshape(dx_s, x.shape)  
  
  #D*M   D*N        N*M
  cdef np.ndarray[np.float64_t, ndim = 2] dw = x_flat.T.dot(dout)
  
  #M*1  M*N           N*1
  cdef np.ndarray[np.float64_t, ndim = 1] db = dout.T.dot(np.ones(N))

  return dx, dw, db

def affine_backward_b(np.ndarray[np.float64_t, ndim = 2] dout, tuple cache):

  x, w, b = cache
  
  # Saving down dimensionality and reshaping
  cdef int N = x.shape[0]
  cdef int D = x.size//N 
 
  cdef np.ndarray[np.float64_t, ndim = 2] x_flat = np.reshape(x, (N, D))
  
  # Propogating derivative through skinny vector format and spreading out
  #N*D  N*M     M*D
  cdef np.ndarray[np.float64_t, ndim = 2] dx_s = dout.dot(w.T)
  
  # Acconting for the case when we are have gotten to the input layer
  cdef np.ndarray[np.float64_t, ndim = 4] dx = np.reshape(dx_s, x.shape)
  
  #D*M   D*N        N*M
  cdef np.ndarray[np.float64_t, ndim = 2] dw = x_flat.T.dot(dout)
  
  #M*1  M*N           N*1
  cdef np.ndarray[np.float64_t, ndim = 1] db = dout.T.dot(np.ones(N))

  return dx, dw, db

def affine_backward(dout, tuple cache):
    """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
    if cache[0].ndim == 2:
        return affine_backward_a(dout, cache)
    if cache[0].ndim == 4:
        return affine_backward_b(dout, cache)

#%%cython

#import numpy as np
#cimport numpy as np
#import cython

#
# Making it leaky
def relu_forward(np.ndarray[np.float64_t, ndim = 2] x, np.float64_t a = 0.1):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """

  cdef np.ndarray[np.float64_t, ndim = 2] cache = x
  
  cdef np.ndarray[np.float64_t, ndim = 2] out = np.maximum(a*x,x) 
  
  return out, cache

#
# Making it leaky
def relu_backward(np.ndarray[np.float64_t, ndim = 2] dout, np.ndarray[np.float64_t, ndim = 2] cache, np.float64_t a = 0.1):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  cdef np.ndarray[np.float64_t, ndim = 2] x = cache
  
  cdef np.ndarray[np.float64_t, ndim = 2] dx = np.copy(dout)
    
  cdef np.ndarray[np.float64_t, ndim = 2] out = 1. * (x > 0)
  out[out == 0] = a  
  
  dx = out * dout  
  
  return dx

#

#%%cython

#import numpy as np
#cimport numpy as np
#import cython
#from __main__ import relu_forward,affine_forward,relu_backward,affine_backward

def affine_relu_forward(x, np.ndarray[np.float64_t, ndim = 2] w, np.ndarray[np.float64_t, ndim = 1] b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache

#

def affine_relu_backward(np.ndarray[np.float64_t, ndim = 2] dout, tuple cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db
  
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