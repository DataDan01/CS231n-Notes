# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:26:34 2017

@author: DA
"""

#%load_ext Cython

#%%cython

#import numpy as np
#cimport numpy as np
#import cython

from numba import jit

@jit 
def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.
  The input x has shape (N, D) and contains a minibatch of N
  examples, where each example x[i] has shape (D). 
  Inputs:
  - x: A numpy array containing input data, of shape (N, D)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns:
  - aff_out: output, of shape (N, M)
  """

  # Computing dot product
  aff_out = np.dot(x, w) + b
  
  return aff_out

# aff_out_test = affine_forward(x = X_train[0:32], w = np.random.randn(3072,100), b = np.random.randn(100)/10)
  
#  
@jit   
def affine_backward(dout, x, w, b):
  """
  Computes the backward pass for an affine layer.
  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - x: Input data, of shape (N, D)
  - w: Weights, of shape (D, M)
  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, D)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  
  N = x.shape[0]
  # Propogating derivative through skinny vector format
  #N*D  N*M     M*D
  dx = dout.dot(w.T)

  #D*M   D*N        N*M
  dw = x.T.dot(dout)
  
  #M*1  M*N           N*1
  db = dout.T.dot(np.ones(N))

  return dx, dw, db

# aff_back_tst = affine_backward(dout = dloss, x = cache['rel4'], w = all_params['W5'], b = all_params['b5'])

@jit 
def relu_forward(x, a):
  """
  Computes the forward pass for a layer of parametric rectified linear units (PReLUs).
  Input:
  - x: Inputs, of any shape (typically the output of an affine layer)
  - a: Leaky ReLU parameters, same length as x
  Returns:
  - relu_out: Output, of the same shape as x
  """
  
  relu_out = np.maximum(0,x) 

  relu_out[relu_out == 0] = (a*x)[relu_out == 0]

  return relu_out

# relu_out_test = relu_forward(x = aff_out_test, a = np.random.randn(100)/20+2/20)

@jit 
# Making it leaky
def relu_backward(dout, aff_out, a):
  """
  Computes the backward pass for a layer of parametric rectified linear units (PReLUs).
  Input:
  - dout: Upstream derivatives, of any shape
  - aff_out: Input x, typically output of previous affine layer
  - a: Leaky ReLU parameters, same length as x
  Returns:
  - dx: Gradient with respect to x
  - da: Gradient with respect to a
  """

  out = 1. * (aff_out > 0)
  
  da = (((1. * (aff_out < 0))*aff_out)*dout).sum(axis=0)

  rep_a = np.array([a]*out.shape[0])

  out[out == 0] += rep_a[out == 0] 
  
  dx = out * dout
   
  return dx, da

# rel_back_tst = relu_backward(dout = dloss, aff_out = cache['aff5'], a = all_params['a5'])

@jit 
def affine_relu_forward(x, w, b, a):
  """
  Convenience layer that performs an affine transform followed by a PReLU
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - a: Parametric ReLU kink parameter
  Returns a tuple of:
  - relu_out: Output from the PReLU
  - aff_out: Ouput from the affine layer
  """
  aff_out = affine_forward(x, w, b)
  
  relu_out= relu_forward(aff_out, a)

  return relu_out, aff_out

# aff_relu_tst = affine_relu_forward(x = X_train[0:32], w = np.random.randn(3072,100), b = np.random.randn(100)/10, a = np.random.randn(100)/20 + 2/20)

#
@jit 
def affine_relu_backward(dout, aff_out, a, x, w, b):
  """
  Backward pass for the affine-relu convenience layer
  """
  
  daff, da = relu_backward(dout, aff_out, a)
  
  dx, dw, db = affine_backward(daff, x, w, b)
  
  return dx, dw, db, da
  
# aff_rel_bk_tst = affine_relu_backward(dout = dloss, aff_out = cache['cache5_aff'], a = all_params['a5'], x = cache['rel5'], w = all_params['W5'], b = all_params['b5'])
 
# 
@jit 
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