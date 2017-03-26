# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:54:30 2017

@author: DA
"""

import numpy as np

def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)

  #############################################################################
  # Implement the Adam update formula, storing the next value of x in         #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #############################################################################
  config['t'] += 1
  config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dx
  config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dx**2)

  # Warm-up phase
  mt = config['m'] / (1-config['beta1']**config['t'])
  vt = config['v'] / (1-config['beta2']**config['t'])

  next_x = x - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon']) 
  
  return next_x, config