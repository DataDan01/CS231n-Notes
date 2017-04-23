# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:26:34 2017

@author: DA
"""
import numpy as np
import im2col as ic

x = X_train[0:10,]
w = np.random.randn(5,3,3,3)
b = np.random.randn(5)

def conv_forward(x, w, b):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  
  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  """
  
  # Adding in zero padding, preserve input size
  # Automatically assuming stride 1 
  npad = (w.shape[3]-1)//2

  x_padded = np.pad(x, 
                    npad, 
                    mode = 'constant', 
                    constant_values = 0)

  x_padded = np.delete(x_padded, np.array([0,x_padded.shape[0]-1]), axis = 0)
  x_padded = np.delete(x_padded, np.array([0,x_padded.shape[3]-1]), axis = 3)

  # Generating im2col
  t = ic.get_im2col_indices(x_shape = x.shape, field_height = 3, field_width = 3, padding=1, stride=1)
  

  return out


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta




#######################################################

import numpy as np
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

  #N*D  N*M     M*D
  dx = dout.dot(w.T)

  #D*M   D*N        N*M
  dw = x.T.dot(dout)
  
  #M*1  M*N           N*1
  db = dout.T.dot(np.ones(N))

  return dx, dw, db

# aff_back_tst = affine_backward(dout = dloss, x = cache['rel4'], w = all_params['W5'], b = all_params['b5'])

@jit 
def relu_forward(x, a, q):
  """
  Computes the forward pass for a layer of parametric rectified linear units (PReLUs).
  Input:
  - x: Inputs, of any shape (typically the output of an affine layer)
  - a: Leaky ReLU parameters for the negative portion, same length as x
  - q: Leaky ReLU parameters for the positive portion, same length as x
  Returns:
  - relu_out: Output, of the same shape as x
  """
  
  # Clipping at zero
  relu_out = np.maximum(0,x) 

  # Positive part
  relu_out[relu_out > 0] = (q*x)[relu_out > 0] 

  # Negative part
  relu_out[relu_out == 0] = (a*x)[relu_out == 0]

  return relu_out

# relu_out_test = relu_forward(x = aff_out_test, a = np.random.randn(100)/50+1/10, q = np.random.randn(100)/20+1)

@jit 
def relu_backward(dout, aff_out, a, q):
  """
  Computes the backward pass for a layer of parametric rectified linear units (PReLUs).
  Input:
  - dout: Upstream derivatives, of any shape
  - aff_out: Output of layer feeding into PReLU, typically affine
  - a: Leaky ReLU parameters for the negative portion
  - q: Leaky ReLU parameters for the positive portion
  Returns:
  - dx: Gradient with respect to x
  - da: Gradient with respect to a
  - dq: Gradient with respect to q
  """

  # Computing derivative on negative portion of PReLU
  da = (((1. * (aff_out < 0))*aff_out)*dout).sum(axis=0)

  # Setup for dq and derivative through positive side of kink
  out = 1. * (aff_out > 0)
  
  dq = (out*aff_out*dout).sum(axis=0)
  
  rep_q = np.array([q]*out.shape[0])
  
  out[out > 0] += rep_q[out > 0]
  
  # Derivative on negative side of kink
  rep_a = np.array([a]*out.shape[0])

  out[out == 0] += rep_a[out == 0] 
  
  dx = out * dout
   
  return dx, da, dq

# rel_back_tst = relu_backward(dout = dloss, aff_out = cache['aff5'], a = all_params['a5'], q = all_params['q5'])

@jit 
def affine_relu_forward(x, w, b, a, q):
  """
  Convenience layer that performs an affine transform followed by a PReLU
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - a: Leaky ReLU parameters, for the negative portion
  - q: Leaky ReLU parameters, for the positive portion
  Returns a tuple of:
  - relu_out: Output from the PReLU
  - aff_out: Ouput from the affine layer
  """
  aff_out = affine_forward(x, w, b)
  
  relu_out= relu_forward(aff_out, a, q)

  return relu_out, aff_out

# aff_relu_tst = affine_relu_forward(x = X_train[0:32], w = np.random.randn(3072,100), b = np.random.randn(100)/10, a = np.random.randn(100)/20 + 2/20, q = np.random.randn(100)/50 + 1)

#
@jit 
def affine_relu_backward(dout, aff_out, a, x, w, b, q):
  """
  Backward pass for the affine-relu convenience layer
  Takes in all inputs into each layer and the intermediate affine output
  """
  
  daff, da, dq = relu_backward(dout, aff_out, a, q)
  
  dx, dw, db = affine_backward(daff, x, w, b)
  
  return dx, dw, db, da, dq
  
# aff_rel_bk_tst = affine_relu_backward(dout = dloss, aff_out = cache['cache5_aff'], a = all_params['a5'], x = cache['rel5'], w = all_params['W5'], b = all_params['b5'])
 
#

@jit
def batchnorm_forward(x, gamma, beta, eps = 1e-8):
  """
  Normalizes inputs to have a mean of zero and standard deviation of one
  Also allows for this operation to be undone or agumented
  Inputs:
  - x: Matrix of data to be normalized N examples * D dimensions
  - gamma: Scale parameter, will stay near 1 if BatchNorm is effective
  - beta: Location parameter, will stay near 0 if BatchNorm is effective
  - eps: Used for numerical stability to avoid division by zero
  Returns:
  - bn_out: Normalized values with beta and gamma applied
  - xhat: Normalized values without beta and gamma applied
  - xmu: Means of inputs
  - var: Variance of inputs
  """
  # Get the dimensions of the input/output  
  N, D = x.shape

  #step1: Calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: Subtract mean vector from every training example
  xmu = x - mu

  #step3: Calculate variance
  var = 1./N * np.sum(xmu**2, axis = 0)

  #step4: Execute normalization
  xhat = xmu * 1./np.sqrt(var + eps)

  #step5: Apply scale parameter
  gammax = gamma * xhat

  #step6: Apply location parameter
  bn_out = gammax + beta

  return bn_out, xhat, xmu, var

# bn_test = batchnorm_forward(x = X_train[0:32], gamma = 1, beta = 0, eps = 1e-8)

#
@jit
def batchnorm_backward(dout, xhat, gamma, xmu, var, eps = 1e-8):
  """
  Computes the backward pass for BatchNorm layer
  Input:
  - dout: Upstream derivatives, of any shape
  - xhat: Normalized values without beta and gamma applied
  - xmu: Means of inputs
  - var: Variance of inputs
  - eps: Used for numerical stability to avoid division by zero
  Returns:
  - dx: Derivative with respect to inputs into BatchNorm layer
  - dgamma: Derivative with respect to scale parameter, gamma
  - dbeta: Derivative with respect to location parameter, beta
  """
  # Get the dimensions of the input/output
  N,D = dout.shape

  # Unpacking var
  sqrtvar = np.sqrt(var + eps)
  ivar = 1./sqrtvar

  #step9
  dbeta = np.sum(dout, axis=0)
  dgammax = dout 

  #step8
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step7
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
  dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
  dx2 = 1. /N * np.ones((N,D)) * dmu

  #step0
  dx = dx1 + dx2

  return dx, dgamma, dbeta

# http://cs231n.github.io/linear-classify/#softmax
# The cross-entropy objective wants the predicted distribution to have all of its mass on the correct answer
# We are minimizing the negative log likelihood of the correct class, maximizing the likelihood of the correct clas (negative * monotonically increasing function)
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
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
                
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

# loss_tst = softmax_loss(x = cache['rel7'], y = y_train[0:32])