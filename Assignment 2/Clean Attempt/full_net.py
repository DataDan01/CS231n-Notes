# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:27:37 2017

@author: DA
"""
import os
os.chdir('C:/Users/Stat-Comp-01/OneDrive/Python/CS231n/Assignment 2/Clean Attempt')

import numpy as np
from numba import jit
import datetime as dt

# Getting the data
exec(open('./data_load.py').read())

# Getting layers and optimization algo
exec(open('./net_layers.py').read())
exec(open('./optim_algs.py').read())

########################################
# Setting up a simple net architecture #
########################################

# Bimodal distribution function
def bimodal(size, mean1 = 1, mean2 = 0.1, sd1 = 0.3, sd2 = 0.1):
    
    def twofer(nothing):
        # Generating random parameters
        a = np.random.normal(mean1,sd1)
        b = np.random.normal(mean2,sd2)
        # Shuffling them around
        if np.random.binomial(1,0.5) == 1:
            return a,b
        else:
            return b,a
    
    twofer_vec = np.vectorize(twofer)
    
    return twofer_vec(np.zeros(size))

# Initializing weights
@jit
def initializer(input_dims = 3072, num_classes = 10, num_layers = 4, layer_width = 128, layer_sd_perc = 0.05, mean1 = 1, mean2 = 0.3, sd1 = 0.25, sd2 = 0.5):
    
    # He initialization, scale for weights and biases
    # Adjusted for the parametric ReLU component
    scale = np.sqrt((2.0)/layer_width) 
     
    # First layer
    
    # Random layer width parameter
    l0_width = int(np.random.normal(layer_width, layer_sd_perc*layer_width))
    
    # Linear, BatchNorm
    all_params = {'W0': np.random.randn(input_dims, l0_width)*scale,
                  'b0': np.random.randn(l0_width)*scale/10.0,
                  'gamma0': np.ones(l0_width),
                  'beta0': np.zeros(l0_width),                 
                  'num_layers': num_layers}
    
    # ReLU init, parameters need to be related to avoid linear behavior
    all_params['a0'],all_params['q0'] = bimodal(l0_width,mean1,mean2,sd1,sd2)
    
    # Middle layers    
    for l in range(1,num_layers-1):
        # Random layer width parameter
        mid_width = int(np.random.normal(layer_width, layer_sd_perc*layer_width))       
        # Linear
        all_params['W'+str(l)] = np.random.randn(all_params['W'+str(l-1)].shape[1], mid_width)*scale
        all_params['b'+str(l)] = np.random.randn(mid_width)*scale/10.0
        # ReLU
        all_params['a'+str(l)],all_params['q'+str(l)] = bimodal(mid_width,mean1,mean2,sd1,sd2)
        # BatchNorm
        all_params['gamma'+str(l)] =  np.ones(mid_width)
        all_params['beta'+str(l)] = np.zeros(mid_width)          
                   
    # Final layer
    # Linear
    all_params['W'+str(num_layers-1)] = np.random.randn(all_params['W'+str(num_layers-2)].shape[1], num_classes)*scale
    all_params['b'+str(num_layers-1)] = np.random.randn(num_classes)*scale/10.0
    # ReLU
    all_params['a'+str(num_layers-1)],all_params['q'+str(num_layers-1)] = bimodal(num_classes,mean1,mean2,sd1,sd2)          
    # BatchNorm
    all_params['gamma'+str(num_layers-1)] =  np.ones(num_classes)
    all_params['beta'+str(num_layers-1)] = np.zeros(num_classes) 
               
    return all_params
                  
# all_params = initializer(input_dims = np.prod(X_train.shape[1:]), num_classes = 10, num_layers = 3, layer_width = 100)

# Forward pass
# x = X_train[0:32]
# y = y_train[0:32]
def forward(all_params, x, y, pred = False):
    
    # Initializing, rel-1 is actually just the input layer
    cache = {'rel-1': x}
    
    # First layer, looking at input explicitly
    # Affine ReLU
    cache['rel0'],cache['aff0'] = affine_relu_forward(x, all_params['W0'], all_params['b0'], all_params['a0'], all_params['q0'])
    
    # BatchNorm after nonlinearity
    cache['bn0'],cache['xhat0'],cache['xmu0'],cache['var0'] = batchnorm_forward(cache['rel0'], all_params['gamma0'], all_params['beta0'])
    
    # Pass through all layers
    for l in range(1,all_params['num_layers']):
    
        layer = str(l)
        prev_lay = str(l-1)
        
        # Affine ReLU
        cache['rel'+layer],cache['aff'+layer] = affine_relu_forward(cache['bn'+prev_lay], all_params['W'+layer], all_params['b'+layer], all_params['a'+layer], all_params['q'+layer])
        
        # BatchNorm after nonlinearity
        cache['bn'+layer],cache['xhat'+layer],cache['xmu'+layer],cache['var'+layer] = batchnorm_forward(cache['rel'+layer], all_params['gamma'+layer], all_params['beta'+layer])
        
    # Prediction time
    if pred == True:
        return np.argmax(cache['bn'+layer], 1)
    
    # Final layer, SoftMax loss
    data_loss, dloss = softmax_loss(cache['bn'+layer], y)
    
    return data_loss, dloss, cache

# data_loss, dloss, cache = forward(all_params = all_params, x = X_train[0:32,], y = y_train[0:32])

# Backward pass
def backward(all_params, dloss, cache, all_configs, learning_rate = 1e-4, reg = 1e-2, beta1 = 0.95, beta2 = 0.99, epsilon = 1e-8, lin_upd_prob = 0.99, rel_upd_prob = 0.10):
    
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
        
        # BatchNorm first
        deriv['drel'],deriv['dgamma'],deriv['dbeta'] = batchnorm_backward(dout = deriv['dx'], xhat = cache['xhat'+layer], gamma = all_params['gamma'+layer], xmu = cache['xmu'+layer], var = cache['var'+layer])
        
        # Updating BatchNorm, always
        all_params['gamma'+layer],all_configs['gamma'+layer] = adam(all_params['gamma'+layer], deriv['dgamma'], all_configs['gamma'+layer])
        all_params['beta'+layer],all_configs['beta'+layer] = adam(all_params['beta'+layer], deriv['dbeta'], all_configs['beta'+layer])
        # Freeing up memory
        deriv['dgamma'] = 0
        deriv['dbeta'] = 0
        cache['xhat'+layer] = 0
        cache['xmu'+layer] = 0 
        cache['var'+layer] = 0
             
        # ReLu --> Affine
        deriv['dx'],deriv['dw'],deriv['db'],deriv['da'],deriv['dq'] = affine_relu_backward(dout = deriv['drel'], aff_out = cache['aff'+layer], a = all_params['a'+layer], x = cache['rel'+prev_lay], w = all_params['W'+layer], b = all_params['b'+layer], q = all_params['q'+layer])
                       
        # Updaing w and b, with probability
        if np.random.binomial(1,lin_upd_prob) == True:
            # Adding in L2 regularization
            deriv['dw'] +=  reg*all_params['W'+layer]**2  
            # Update
            all_params['W'+layer],all_configs['W'+layer] = adam(all_params['W'+layer], deriv['dw'],all_configs['W'+layer])
            all_params['b'+layer],all_configs['b'+layer] = adam(all_params['b'+layer], deriv['db'],all_configs['b'+layer])
        # Freeing up memory
        deriv['dw'] = 0
        deriv['db'] = 0
        cache['aff'+layer] = 0
        cache['rel'+prev_lay] = 0
             
        # Updaing a and q, with probability
        if np.random.binomial(1,rel_upd_prob) == True:
            # Update
            all_params['a'+layer],all_configs['a'+layer] = adam(all_params['a'+layer], deriv['da'],all_configs['a'+layer])
            all_params['q'+layer],all_configs['q'+layer] = adam(all_params['q'+layer], deriv['dq'],all_configs['q'+layer])
        # Freeing up memory
        deriv['da'] = 0
        deriv['dq'] = 0
             
    return all_params, all_configs

# all_params, all_configs = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = None, learning_rate = 1e-4, reg = 1e-4, beta1 = 0.95, beta2 = 0.999, epsilon = 1e-8)
  
# Prediction 
# preds = forward(all_params, X_train[0:200,], y = None, pred = True)

#
def training(all_params, all_configs, X, y, X_val, y_val, num_layers, layer_width, layer_sd_perc = 0.05, mean1 = 1, mean2 = 0.3, sd1 = 0.25, sd2 = 0.5, batch_size = 32, niter = 10000, init_lr = 1e-4, reg = 0, beta1 = 0.95, beta2 = 0.99, print_every = 100, check_every = 100, break_when = 10, break_perc = 0.05, lin_upd_prob = 0.99, rel_upd_prob = 0.10):
   
   # Initialize parameters if there are none
   if all_params == None:
       all_params = initializer(input_dims = np.prod(X.shape[1:]), num_classes = len(np.unique(y)), num_layers = num_layers, layer_width = layer_width, layer_sd_perc = layer_sd_perc, mean1 = mean1, mean2 = mean2, sd1 = sd1, sd2 = sd2)

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
       
       # Parameter update
       globals()['all_params'], globals()['all_configs'] = backward(all_params = all_params, dloss = dloss, cache = cache, all_configs = all_configs, learning_rate = init_lr, reg = reg, beta1 = beta1, beta2 = beta2, epsilon = 1e-8, lin_upd_prob = lin_upd_prob, rel_upd_prob = rel_upd_prob)
       
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
               # Saving down best accuracy and time
               np.save(file = 'C:/Users/Stat-Comp-01/OneDrive/Python/CS231n/Assignment 2/Clean Attempt/Accs/'+str(round(globals()['best_acc']*100,2))+'_acc', arr = np.array([0]))
               globals()['best_time'] = dt.datetime.now()
           # Keep track of weak improvement
           if acc <= globals()['best_acc']:
               break_counter += 1
               # Break training when improvement is too slow or when current accuracy is too far away
               if break_counter == break_when or acc+break_perc <= globals()['best_acc']:
                   print('Stopping early')
                   break   
   return None

##

#all_params,all_configs,best_acc = None, None, 0
  
# training(all_params = all_params, all_configs = all_configs, X = X_train, y = y_train, X_val = X_val, y_val = y_val, num_layers = 20, layer_width = 1024, batch_size = 128, niter = int(1e4), init_lr = 1e-4, reg = 1e-4, beta1 = 0.99, beta2 = 0.99, print_every = 5, check_every = 5, break_when = 20)

# Overfitting small subset to test out model
# training(all_params = all_params, all_configs = all_configs, X = X_train[0:10,], y = y_train[0:10], X_val = X_train[0:10,], y_val = y_train[0:10], num_layers = 10, layer_width = 100, batch_size = 10, niter = int(1e4), init_lr = 1e-4, reg = 0, beta1 = 0.95, beta2 = 0.95, print_every = 100, check_every =  100)

# Element wise injection of noise, not too big
small_noise = np.vectorize(lambda x: np.random.normal(0,abs(x)/50) + x)

best_acc, best_time = 0, dt.datetime.now()

# First learing rate and exponential decay of rate
glob_lr = 1e-3
dec_rate = 0.8

for i in range(10000):
    
    # Wipe out configurations
    all_configs = None
    
    # Reading in parameters if they aren't in memory
    if ('all_params' in globals().keys()) == False:
        all_params = {}
        for file in os.listdir('D:/Py_Data/'):
            all_params[re.sub('.npy','',file)] = np.load('D:/Py_Data/'+file)
    
    # If the data directory is empty and/or no data have been read in, create new parameters from scratch
    if len(all_params.keys()) == 0:
        all_params = None
    
    print('Restarting training')
    
    training(all_params = all_params, 
             all_configs = all_configs, 
             X = X_train, 
             y = y_train, 
             X_val = X_val, 
             y_val = y_val, 
             num_layers = 15, 
             layer_width = 4096,
             layer_sd_perc = 0.05, 
             mean1 = 1, 
             mean2 = 0.1, 
             sd1 = 0.2, 
             sd2 = 0.2, 
             batch_size = 256+2*i, 
             niter = int(1e4), 
             init_lr = glob_lr, 
             reg = np.random.uniform(1e-2,1e-8), 
             beta1 = np.random.uniform(0.5,1-1e-4), 
             beta2 = np.random.uniform(0.90,1-1e-4), 
             print_every = 1, 
             check_every = 1, 
             break_when = 50, 
             break_perc = 0.015,
             lin_upd_prob = 0.99, 
             rel_upd_prob = 0.10)

    # Decaying the learning rate after break or completion of training round
    glob_lr *= dec_rate 

    # Figuring out when the last update was and adding noise if it was a while ago
    if dt.datetime.now() > best_time + dt.timedelta(minutes=30):
        print('Adding noise to parameters, resetting best accuracy')
        # Resetting best accuracy
        best_acc = 0
        np.save(file = 'C:/Users/Stat-Comp-01/OneDrive/Python/CS231n/Assignment 2/Clean Attempt/Accs/RESET', arr = np.array([0]))
        # Increasing learning rate a bit to move out of lower accuracy faster
        glob_lr *= 1/(dec_rate**8)
        # Injecting noise into all parameters
        for key in all_params:
            if key == 'num_layers':
                continue
            all_params[key] = small_noise(all_params[key])

    
# Finding out total size
size = 0

for key in all_params:
    if isinstance(all_params[key], np.ndarray):
        size += all_params[key].size
    
print ('{:,} parameters totaling {:,.0f} MB'.format(size,size*64/(8*10**6))) 
