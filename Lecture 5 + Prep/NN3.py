# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:07:35 2017

@author: DA
"""

# assume parameter vector W and its gradient vector dW
param_scale = np.linalg.norm(W.ravel())
update = -learning_rate*dW # simple SGD update
update_scale = np.linalg.norm(update.ravel())
W += update # the actual update
print(update_scale / param_scale) # want ~1e-3

# Vanilla update
x += - learning_rate * dx

# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position

# Nesterov Momentum
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v

v_prev = v # back this up
v = mu * v - learning_rate * dx # velocity update stays the same
x += -mu * v_prev + (1 + mu) * v # position update changes form

# Adagrad
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)

# RMSProp
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)

