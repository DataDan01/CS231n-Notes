# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:27:37 2017

@author: DA
"""
import os
os.chdir('C:/Users/Stat-Comp-01/OneDrive/Python/CS231n/Assignment 2/Clean Attempt')

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
@jit
def initializer(input_dims, num_classes, num_layers, layer_width, scale):
    
    if scale == None:
        scale = np.sqrt(2.0/layer_width) # He initialization
     
    # Initialization
    all_params = {'W0': np.random.randn(input_dims, layer_width)*scale,
                  'b0': np.random.randn(layer_width)*scale/10.0,
                  'a0': np.random.randn(layer_width)*scale/20.0 + 2/20,
                  'num_layers': num_layers,
                  'layer_width': layer_width}
    
    # Middle layers    
    for l in range(1,num_layers-1):
        all_params['W'+str(l)] = np.random.randn(layer_width, layer_width)*scale
        all_params['b'+str(l)] = np.random.randn(layer_width)*scale/10.0
        all_params['a'+str(l)] = np.random.randn(layer_width)*scale/20.0 + 2/20
                   
    # Final layer
    all_params['W'+str(num_layers-1)] = np.random.randn(layer_width, num_classes)*scale
    all_params['b'+str(num_layers-1)] = np.random.randn(num_classes)*scale/10.0
    all_params['a'+str(num_layers-1)] = np.random.randn(num_classes)*scale/20.0 + 2/20           
               
    return all_params
                  
# all_params = initializer(input_dims = np.prod(X_train.shape[1:]), num_classes = 10, num_layers = 5, layer_width = 100, scale = 1e-6)

# Forward pass
# x = X_train[0:32]
# y = y_train[0:32]
def forward(all_params, x, y, pred = False):
    
    # Initializing, rel-1 is actually just the input layer
    cache = {'rel-1': x}
    
    # First layer, looking at input explicitly
    cache['rel0'],cache['aff0'] = affine_relu_forward(x, all_params['W0'], all_params['b0'], all_params['a0'])
    
    # Pass through all layers
    for l in range(1,all_params['num_layers']):
    
        layer = str(l)
        prev_lay = str(l-1)

        cache['rel'+layer],cache['aff'+layer] = affine_relu_forward(cache['rel'+prev_lay], all_params['W'+layer], all_params['b'+layer], all_params['a'+layer])
        
    # Prediction time
    if pred == True:
        return np.argmax(cache['rel'+layer], 1)
    
    # Final layer, SoftMax loss
    data_loss, dloss = softmax_loss(cache['rel'+layer], y)
    
    return data_loss, dloss, cache

# data_loss, dloss, cache = forward(all_params, x = X_train[0:32,], y = y_train[0:32])

# Backward pass
def backward(all_params, dloss, cache, all_configs, learning_rate, reg, beta1, beta2, epsilon):
    
    # Setup to save/initialize configurations and derivatives
    if all_configs == None:
        
        all_configs = {}
        
        for key in all_params:        
        
            if key in ['num_layers','layer_width']:
                continue
            
            all_configs[key] = {
                    'learning_rate': learning_rate,
                    'beta1': beta1,
                    'beta2': beta2,
                    'epsilon': epsilon,
                    'm': np.zeros_like(all_params[key]),
                    'v': np.zeros_like(all_params[key]),
                    't': 0 }       
            
    deriv = {'dx': dloss}    
        
    # Middle layers
    for i in range(all_params['num_layers']-1, -1, -1):
                
        layer = str(i)
        prev_lay = str(i-1)
        
        deriv['dx'],deriv['dw'],deriv['db'],deriv['da'] = affine_relu_backward(dout = deriv['dx'], aff_out = cache['aff'+layer], a = all_params['a'+layer], x = cache['rel'+prev_lay], w = all_params['W'+layer], b = all_params['b'+layer])
        
        # Adding in L2 regularization
        deriv['dw'] +=  reg*all_params['W'+layer]**2   
        
        # Updating all parameters and freeing up memory
        cache['aff'+layer] = 0
        cache['rel'+prev_lay] = 0
        
        # Updaing w
        all_params['W'+layer],all_configs['W'+layer] = adam(all_params['W'+layer], deriv['dw'],all_configs['W'+layer])
        deriv['dw'] = 0
             
        # Updating b
        all_params['b'+layer],all_configs['b'+layer] = adam(all_params['b'+layer], deriv['db'],all_configs['b'+layer])
        deriv['db'] = 0
             
        # Updaing a
        all_params['a'+layer],all_configs['a'+layer] = adam(all_params['a'+layer], deriv['da'],all_configs['a'+layer])
        deriv['da'] = 0
                
    return all_params, all_configs

# all_params, all_configs = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = None, learning_rate = 1e-4, reg = 1e-4, beta1 = 0.95, beta2 = 0.999, epsilon = 1e-8)
  
# all_params, all_configs = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = all_configs, learning_rate = 1e-4, reg = 1e-4,beta1 = 0.95, beta2 = 0.999, epsilon = 1e-8)

# Prediction 
# preds = forward(all_params, X_train[0:200,], y = None, pred = True)

#
def training(all_params, all_configs, X, y, X_val, y_val, num_layers, layer_width, scale, batch_size, niter, init_lr, reg, beta1, beta2, print_every = 100, check_every = 100, break_when = 10):
   
   # Initialize parameters if there are none
   if all_params == None:
       all_params = initializer(input_dims = np.prod(X.shape[1:]), num_classes = len(np.unique(y)), num_layers = num_layers, layer_width = layer_width, scale = scale)

   # Initializing accumulation variable for early stopping
   break_counter = 1

   # Forward and backward passes through all layers
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
       
       # Displaying validation accuracy and progress at batch intervals
       if i % print_every == 0:
           val_preds = forward(all_params, X_val, y = None, pred = True)
           
           acc = np.mean(val_preds == y_val)
           
           print('Data loss {:.3f}, Val acc {:.2f}%, Iter {:.2f}%, Break {}/{}'.format(data_loss,acc*100,(i+print_every)/niter*100,break_counter,break_when))
       
       # Keep track of how well the model is learning       
       if i % check_every == 0 and i > 1:
           # Update best accuracy when record is beat, write data and reset break counter
           if acc > globals()['best_acc']:
               globals()['best_acc'] = acc
               break_counter = 0       
               for key in all_params:
                   np.save(file = 'D:/Py_Data/'+str(key), arr = all_params[key])
               # Saving down best accuracy
               np.save(file = 'C:/Users/Stat-Comp-01/OneDrive/Python/CS231n/Assignment 2/Clean Attempt/Accs/'+str(round(globals()['best_acc']*100,2))+'_acc', arr = np.array([0]))
           # Keep track of weak improvement
           if acc <= globals()['best_acc']:
               break_counter += 1
               # Break training when improvement is too slow or when current accuracy is too far away
               if break_counter == break_when or acc+0.05 <= globals()['best_acc']:
                   print('Stopping early')
                   break   
   return None

##

#all_params,all_configs,best_acc = None, None, 0
  
#training(all_params = all_params, all_configs = all_configs, X = X_train, y = y_train, X_val = X_val, y_val = y_val, num_layers = 20, layer_width = 1024, scale = None, batch_size = 128, niter = int(1e4), init_lr = 1e-4, reg = 1e-4, beta1 = 0.99, beta2 = 0.99, print_every = 5, check_every = 5, break_when = 20)

# Overfitting small subset to test out model
# training(all_params = all_params, all_configs = all_configs, X = X_train[0:10,], y = y_train[0:10], X_val = X_train[0:10,], y_val = y_train[0:10], num_layers = 10, layer_width = 100, scale = None, batch_size = 10, niter = int(1e4), init_lr = 1e-4, reg = 0, beta1 = 0.95, beta2 = 0.95, print_every = 100, check_every =  100)

import re
import os

best_acc = 0

for i in range(10000):
    
    # Read parameters in if they are saved on disk
    all_params,all_configs = {},None
    
    # Reading in parameters
    for file in os.listdir('D:/Py_Data/'):
        all_params[re.sub('.npy','',file)] = np.load('D:/Py_Data/'+file)
    
    # If the data directory is empty, create new parameters from scratch
    if len(all_params.keys()) == 0:
        all_params = None
    
    print('Reloaded data, starting training')
    
    training(all_params = all_params, 
             all_configs = all_configs, 
             X = X_train, 
             y = y_train, 
             X_val = X_val, 
             y_val = y_val, 
             num_layers = 8, 
             layer_width = 512, 
             scale = None, 
             batch_size = 1024, 
             niter = int(1e4), 
             init_lr = np.random.uniform(1e-3,1e-8), 
             reg = np.random.uniform(1e-1,1e-8), 
             beta1 = np.random.uniform(0.5,1-1e-4), 
             beta2 = np.random.uniform(0.95,1-1e-4), 
             print_every = 1, 
             check_every = 1, 
             break_when = 50)

# Finding out total size
size = 0

for key in all_params:
    if isinstance(all_params[key], np.ndarray):
        size += all_params[key].size
    
print ('{:,} parameters totaling {:,.0f} MB'.format(size,size*64/(8*10**6))) 
