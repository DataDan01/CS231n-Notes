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
 
def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.
  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.
  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns:
  - aff_out: output, of shape (N, M)
  """
  
  # Reshaping the input into a batch of examples as row vectors
  N = x.shape[0]
  D = np.prod(x.shape[1:])  
  
  x_flat = np.reshape(x, (N, D))
  
  # Computing dot product
  aff_out = np.dot(x_flat, w) + b
  
  return aff_out

# aff_out_test = affine_forward(x = X_train[0:32], w = np.random.randn(3072,100), b = np.random.randn(100)/10)
  
#  
  
def affine_backward(dout, x, w, b):
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
  
  # Saving down dimensionality and reshaping
  N = x.shape[0]
  D = np.prod(x.shape[1:])    

  # Propogating derivative through skinny vector format and spreading out
  #N*D  N*M     M*D
  dx = dout.dot(w.T)
  
  #D*M   D*N        N*M
  dw = x.T.dot(dout)
  
  #M*1  M*N           N*1
  db = dout.T.dot(np.ones(N))

  return dx, dw, db

# aff_back_tst = affine_backward(dout = dloss, x = cache['cache5_rel'], w = all_params['W5'], b = all_params['b5'])

#
# Making it leaky
def relu_forward(x, a):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).
  Input:
  - x: Inputs, of any shape (typically the output of an affine layer)
  - a: Leaky ReLU parameters, same length as x
  Returns a tuple of:
  - relu_out: Output, of the same shape as x
  """
  
  relu_out = np.maximum(a*x,x) 

  return relu_out

# relu_out_test = relu_forward(x = aff_out_test, a = np.random.randn(100)/10)

#
# Making it leaky
def relu_backward(dout, cache, a):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).
  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout
  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  
  dx = np.copy(dout)
    
  out = 1. * (x > 0)
  out[out == 0] = a  
  
  dx = out * dout  
  
  return dx

#

def affine_relu_forward(x, w, b, a):
  """
  Convenience layer that perorms an affine transform followed by a ReLU
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - a: Parametric ReLU kink parameter
  Returns a tuple of:
  - relu_out: Output from the ReLU
  - aff_out: Ouput from the affine layer
  """
  aff_out = affine_forward(x, w, b)
  
  relu_out= relu_forward(aff_out, a)

  return relu_out, aff_out

# aff_relu_tst = affine_relu_forward(x = X_train[0:32], w = np.random.randn(3072,100), b = np.random.randn(100)/10, a = np.random.randn(100)/10)

#

def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db
  
 
# 

def softmax_loss(x, y):
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
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  
  N = x.shape[0]
  
  loss = -np.sum(np.log(probs[np.arange(N), y]))/N
  
  dx = probs.copy()
  dx[np.arange(N), y] -= 1.0
  dx /= N
  
  return loss, dx