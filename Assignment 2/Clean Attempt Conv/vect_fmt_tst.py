# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:03:41 2017

@author: Stat-Comp-01
"""

import os
import numpy as np

os.chdir('C:/Users/Stat-Comp-01/OneDrive/Python/CS231n/Assignment 2/Clean Attempt Conv')

# Getting the data
exec(open('./data_load.py').read())

x = X_train[0:10,]
w = np.random.randn(5,3,3,3)

npad = (w.shape[3]-1)//2

x_padded = np.pad(x, 
                  npad, 
                  mode = 'constant', 
                  constant_values = 0)

x_padded = np.delete(x_padded, np.array([0,x_padded.shape[0]-1]), axis = 0)
x_padded = np.delete(x_padded, np.array([0,x_padded.shape[3]-1]), axis = 3)


# Testing
x_skinny_raw = np.reshape(X_train[0],np.prod(X_train[0].shape))
x_skinny_padded = np.reshape(x_padded[0],np.prod(x_padded[0].shape))

x_fat_raw = X_train[0]
x_fat_padded = x_padded[0]

# Need a systematic way to convert each image to a zero padded column
# Then create index for filter to pass over

orig_dim = X_train[0].shape[1] 
depth = X_train[0].shape[2]

arr = np.reshape(X_train[0], np.prod(X_train[0].shape))

def pad_filter(arr, orig_dim, depth, filter_size = 3):
    """
  Figures out how to correctly pad a numpy array with zeros and pass a filter over it
  Input:
      - arr: Numpy vector, skinny one dimensional version of a volume
      - orig_dim: Original height & width of the volume
      - depth: Original depth of the volume
      - filter_size: The size of the filer passed over the image
  Output: 
      - pad_vect: The padded vector with all of the 0s in the correct places
      - vect_map: The map that contains index locations of the filter moving across the padded image
    """
    ###############
    ### Padding ###
    ###############
    # Determing amount of padding needed to maintain size
    npad = (filter_size-1)//2
    
    # Step (1) Pad beginning with zeros, used to also pad bottom
    top_bot = np.zeros(npad*depth*(2*npad+orig_dim))

    # Step (2) Chop vector into pieces, pad each piece
    chunks = np.split(arr, orig_dim)
    
    chunks = [np.concatenate([np.zeros(npad*depth),
                              x,
                              np.zeros(npad*depth)]) for x in chunks]
    
    # Step (3) combine top and bottom padding with middle values, reformat
    padded_vect = np.concatenate([top_bot,
                                  np.concatenate(chunks),
                                  top_bot])
    #################
    ### Filtering ###
    #################
    # The "filter" will just be an list of index locations along the padded vect
    new_dim = orig_dim+2*npad

    # Upper left corner
    left = np.array([new_dim*i*depth for i in range(filter_size)],
                     dtype = np.uint32) # Always going to be +, change to larger size
                                    # if more than 4294967295 out_size 
    
    left_full = np.concatenate([left + i for i in range(depth)])
    
    # Expanding out the full box    
    box = np.concatenate([left_full + i*depth for i in range(filter_size)])

    # Sliding the box
    first_row = np.concatenate([box + i*depth for i in range(orig_dim)])    

    vect_map = np.concatenate([first_row + i*new_dim*depth for i in range(orig_dim)])
    
    return padded_vect, vect_map

# Testing
test_pad,test_map = pad_filter(np.reshape(X_train[0],3072), 
                               orig_dim = 32, 
                               depth = 3, 
                               filter_size = 3)

t = test_pad[test_map]

test_pad.nbytes+test_map.nbytes # 138336

t.nbytes # 221184

# Setting up convolution function that adheres to format above

filter_w = np.random.randn(3**2)

def convolve(padded_vect, vect_map, filter_w):
    """
  Applies convolution over padded vector with filter map
  Input:
      - padded_vect: Skinny version of image volume with zeroes in the correct places for the filter to work
      - vect_map: Array of indices that represents a filter passing over the vect_map
      - filter_w: Minial set of weights to represent the filter
  Output: 
      - new_volume: Matrix that represents various filters passing over the input volume
    """
    
    
    
    return 1