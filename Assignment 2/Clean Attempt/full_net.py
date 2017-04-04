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
        scale = np.sqrt(2.0/layer_width) # He initialization
     
    # First layer
    all_params = {'W0': np.random.randn(input_dims, layer_width)*scale,
                  'b0': np.random.randn(layer_width)*scale/10.0,
                  'a0': np.random.randn(layer_width)*scale/100.0,
                  'num_layers': num_layers,
                  'layer_width': layer_width}
    
    # Initialize middle layers    
    all_params['W_mid'] = np.zeros((layer_width,layer_width,num_layers-1), dtype = np.float64)   
    all_params['b_mid'] = np.zeros((layer_width,num_layers-1), dtype = np.float64)
    all_params['a_mid'] = np.zeros((layer_width,num_layers-1), dtype = np.float64)
    
    # Middle Layers
    for l in range(num_layers-1):
        all_params['W_mid'][:,:,l] = np.random.randn(layer_width, layer_width)*scale
        all_params['b_mid'][:,l] = np.random.randn(layer_width)*scale/10.0
        all_params['a_mid'][:,l] = np.random.randn(layer_width)*scale/100.0
        
    # Final layer
    all_params['W'+str(num_layers)] = np.random.randn(layer_width, num_classes)*scale
    all_params['b'+str(num_layers)] = np.random.randn(num_classes)*scale/10.0
    all_params['a'+str(num_layers)] = np.random.randn(num_classes)*scale/100.0
        
    return all_params
                  
# all_params = initializer(input_dims = np.prod(X_train.shape[1:]), num_classes = 10, num_layers = 5, layer_width = 100, scale = 1e-6)

# Forward pass
# x = X_train[0:32]
# y = y_train[0:32]
def forward(all_params, x, y, pred = False):

    # Initializing all cache parameters for a single pass
    last_layer = str(all_params['num_layers'])
    
    batch_size = x.shape[0]    
    
    # We only need all of the intermediate layer outputs, we have everything else
    cache = {'cache_mid_rel': np.zeros((batch_size,all_params['W_mid'].shape[1],all_params['W_mid'].shape[2])),
             'cache'+last_layer+'_rel': np.zeros_like(all_params['W'+last_layer]),
             'cache_mid_aff': np.zeros((batch_size,all_params['W_mid'].shape[1],all_params['W_mid'].shape[2])),
             'cache'+last_layer+'_aff': np.zeros_like(all_params['W'+last_layer])
}


    # Accounting for first layer      
    cache['cache0_rel'],cache['cache0_aff'] = affine_relu_forward(x, all_params['W0'], all_params['b0'], all_params['a0'])

    # First Middle Layer
    cache['cache_mid_rel'][:,:,1],cache['cache_mid_aff'][:,:,1] = affine_relu_forward(cache['cache0_rel'], all_params['W_mid'][:,:,1], all_params['b_mid'][:,1], all_params['a_mid'][:,1])

    # Pass through all layers
    for l in range(2,all_params['num_layers']-1):
    
        # Middle layers
        cache['cache_mid_rel'][:,:,l],cache['cache_mid_aff'][:,:,l] = affine_relu_forward(cache['cache_mid_rel'][:,:,l-2], all_params['W_mid'][:,:,l], all_params['b_mid'][:,l], all_params['a_mid'][:,l])
        
    # Final Layer
    cache['cache'+last_layer+'_rel'],cache['cache'+last_layer+'_aff'] = affine_relu_forward(cache['cache_mid_aff'][:,:,int(last_layer)-2], all_params['W'+last_layer], all_params['b'+last_layer], all_params['a'+last_layer])
        
    # Prediction time
    if pred == True:
        return np.argmax(final_output, 1)
    
    # Final layer, SoftMax loss
    data_loss, dloss = softmax_loss(cache['cache'+last_layer+'_rel'], y)
    
    return data_loss, dloss, cache

# data_loss, dloss, cache = forward(all_params, x = X_train[0:32,], y = y_train[0:32])

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
def training(all_params, all_configs, X, y, X_val, y_val, num_layers, layer_width, scale, batch_size, niter, init_lr, reg, beta1, beta2, print_every = 100):
   
   # Initialize parameters if there are none
   if all_params == None:
       all_params = initializer(input_dims = np.prod(X.shape[1:]), num_classes = len(np.unique(y)), num_layers = num_layers, layer_width = layer_width, scale = scale)

   # Subsequent passes
   for i in range(niter):
       
       # Batch setup
       rand_inx = np.random.randint(0, X.shape[0], batch_size)    
       
       X_mini = X[rand_inx,]
       y_mini = y[rand_inx] 
           
       # Forward pass to get loss    
       data_loss, dloss, cache = forward(all_params, X_mini, y_mini)
       
       # Decaying the learning rate linearly
       dec_lr = init_lr*(1-i/niter)
       
       # Parameter update
       globals()['all_params'], globals()['all_configs'] = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = all_configs, learning_rate = dec_lr, reg = reg, beta1 = beta1, beta2 = beta2, epsilon = 1e-8)

       # Displaying validation accuracy and progress at 100 batch intervals
       if i % print_every == 0:
           val_preds = forward(all_params, X_val, y = None, pred = True)
           
           acc = np.mean(val_preds == y_val)
           
           print('Data loss {:.3f}, Val accuracy {:.2f}%, Iter {:.2f}%'.format(data_loss,acc*100,(i+print_every)/niter*100))
                     
   return all_params, all_configs

##

all_params,all_configs = None, None
  
all_params,all_configs = training(all_params = all_params, all_configs = all_configs, X = X_train, y = y_train, X_val = X_val, y_val = y_val, num_layers = 5, layer_width = 128, scale = None, batch_size = 256, niter = int(1e4), init_lr = 1e-4, reg = 1e-8, beta1 = 0.99, beta2 = 0.99, print_every = 1)

# Overfitting small subset to test out model
# all_params,all_configs = training(all_params = all_params, all_configs = all_configs, X = X_train[0:50,], y = y_train[0:50], X_val = X_train[0:50,], y_val = y_val[0:50], num_layers = 5, layer_width = 100, scale = None, batch_size = 32, niter = int(1e4), init_lr = 1e-4, reg = 0, beta1 = 0.95, beta2 = 0.95, print_every = 100)