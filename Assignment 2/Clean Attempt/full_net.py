# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:27:37 2017

@author: DA
"""
# import os
# os.chdir('./OneDrive/Python/CS231n/Assignment 2/Clean Attempt')

import numpy as np

from numba import jit

# Getting the data
exec(open('./data_load.py').read())

# Getting layers and optimization algo
exec(open('./net_layers.py').read())
exec(open('./optim_algs.py').read())

########################################
# Setting up a simple net architecture #
########################################

# Initializing weights
#@jit
def initializer(input_dims, num_classes, num_layers, layer_width, scale):
    
    if scale == None:
        scale = 1/np.sqrt(layer_width)
    
    # First layer
    all_params = {'W0': np.random.randn(input_dims, layer_width)*scale,
                  'b0': np.zeros(layer_width) }
    
    # Middle Layers
    for l in range(num_layers-1):
        all_params['W'+str(l+1)] = np.random.randn(layer_width, layer_width)*scale
        all_params['b'+str(l+1)] = np.zeros(layer_width)
        
    # Final layer
    all_params['W'+str(num_layers)] = np.random.randn(layer_width, num_classes)*scale
    all_params['b'+str(num_layers)] = np.zeros(num_classes)
        
    return all_params
                  
# all_params = initializer(input_dims = np.prod(X_train.shape[1:]), num_classes = 10, num_layers = 2, layer_width = 100, scale = 1e-6)

# Forward pass
#@jit
def forward(all_params, x, y, pred = False):

    cache = {}

    next_input = x

    # First and all middle layers
    for l in range(len(all_params)//2):
        
        str_inx = str(l)        
        
        current_cache = 'cache'+str_inx
        current_weights = 'W'+str_inx       
        current_biases = 'b'+str_inx
        
        next_input, cache[current_cache] = affine_relu_forward(next_input, all_params[current_weights], all_params[current_biases])
    
    final_output = next_input 
    
    # Prediction time
    if pred == True:
        return np.argmax(final_output, 1)
    
    # Final layer, SoftMax loss
    data_loss, dloss = softmax_loss(final_output, y)
    
    return data_loss, dloss, cache

# data_loss, dloss, cache = forward(all_params, X_train[0:200,], y_train[0:200])

# Backward pass
#@jit
def backward(all_params, dloss, cache, all_configs, learning_rate, reg, beta1, beta2, epsilon):
    
    # Setup to save/initialize configurations and derivatives
    if all_configs == None:
        
        all_configs = {}
        
        for key in all_params:        
        
            all_configs[key] = {
                    'learning_rate': learning_rate,
                    'beta1': beta1,
                    'beta2': beta2,
                    'epsilon': epsilon,
                    'm': np.zeros_like(all_params[key]),
                    'v': np.zeros_like(all_params[key]),
                    't': 0 }        
            
    deriv = {'dx': dloss}    
        
    for i in range(len(all_params)//2-1, -1, -1):
        
        # Setting up references for indexing dicts
        str_inx = str(i)        
        
        current_w = 'W'+str_inx
        current_b = 'b'+str_inx
        current_cache = 'cache'+str_inx
        last_cache = 'cache'+str(int(str_inx)+1)
        
        # Computing raw backprop gradient
        deriv['dx'],deriv['dw'],deriv['db'] = affine_relu_backward(deriv['dx'], cache[current_cache])
        
        # Adding in L2 regularization
        deriv['dw'] +=  reg*all_params[current_w]**2       
        
        # Applying adam to weights and biases
        all_params[current_w],all_configs[current_w] = adam(all_params[current_w], deriv['dw'],all_configs[current_w])
        
        all_params[current_b],all_configs[current_b] = adam(all_params[current_b], deriv['db'],all_configs[current_b])
        
        # Clearing up some memory
        if i == len(all_params)//2:
            continue
        cache[last_cache] = 0
        
        if i == 0:
            cache['cache0'] = 0
        
    return all_params, all_configs

# Init
# all_params = initializer(input_dims = np.prod(X_train.shape[1:]), num_classes = 10, num_layers = 2, layer_width = 100, scale = 1e-3)

# First Passes
# data_loss, dloss, cache = forward(all_params, X_train[0:200,], y_train[0:200])    
# all_params, all_configs = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = None, learning_rate = 1e-4, beta1 = 0.95, beta2 = 0.999, epsilon = 1e-8)

# Subsequent Passes
# data_loss, dloss, cache = forward(all_params, X_train[0:200,], y_train[0:200])    
# all_params, all_configs = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = all_configs, learning_rate = 1e-4, beta1 = 0.95, beta2 = 0.999, epsilon = 1e-8)

# Prediction 
# preds = forward(all_params, X_train[0:200,], y = None, pred = True)

#@jit
def training(all_params, X, y, X_val, y_val, num_layers, layer_width, scale, batch_size, niter, init_lr, reg, beta1, beta2, print_every = 100):
   
   # Initialize parameters if there are none
   if all_params == None:
       all_params = initializer(input_dims = np.prod(X.shape[1:]), num_classes = len(np.unique(y)), num_layers = num_layers, layer_width = layer_width, scale = scale)

   # Subsequent passes
   for i in range(niter):
       
       # Batch setup
       rand_inx = np.random.randint(0, X.shape[0], batch_size)    
       
       X_mini = X[rand_inx,]
       y_mini = y[rand_inx] 
           
       # First pass
       if i == 0:
           data_loss, dloss, cache = forward(all_params, X_mini, y_mini)          
           
           all_params, all_configs = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = None, learning_rate = init_lr, reg = reg, beta1 = beta1, beta2 = beta2, epsilon = 1e-8)
       # Subsequent passes    
       data_loss, dloss, cache = forward(all_params, X_mini, y_mini)
       
       # Decaying the learning rate linearly
       dec_lr = init_lr*(1-i/niter)
       
       try:
           all_params, all_configs = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = all_configs, learning_rate = dec_lr, reg = reg, beta1 = beta1, beta2 = beta2, epsilon = 1e-8)
       # Ctrl + C to stop training but still update parameters
       except KeyboardInterrupt:
           return all_params
       
       # Displaying validation accuracy and progress at 100 batch intervals
       if i % print_every == 0:
           val_preds = forward(all_params, X_val, y = None, pred = True)
           
           acc = np.mean(val_preds == y_val)
           
           print('Data loss {:.3f}, Val accuracy {:.2f}%, Iter {:.2f}%'.format(data_loss,acc*100,(i+print_every)/niter*100))
           
   return all_params

##

all_params = None
  
all_params = training(all_params = all_params, X = X_train, y = y_train, X_val = X_val, y_val = y_val, num_layers = 5, layer_width = 2048, scale = 1e-3, batch_size = 256, niter = int(1e4), init_lr = 1e-4, reg = 1e-8, beta1 = 0.95, beta2 = 0.999, print_every = 10)

# all_params = training(all_params = all_params, X = X_train[0:100,], y = y_train[0:100], X_val = X_val[0:100,], y_val = y_val[0:100], num_layers = 2, layer_width = 20, scale = 1e-3, batch_size = 2, niter = int(1e4), init_lr = 1e-3, reg = 0, beta1 = 0.95, beta2 = 0.999, print_every = 100)