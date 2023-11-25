# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:26:47 2023

@author: khrstln
"""
import numpy as np
import scipy

def Get_next_x(f_noise: np.array, K: np.array, x_prev: np.array, alpha: float, N: int):
    return x_prev[:N] + alpha * (f_noise[:N] - scipy.signal.convolve(x_prev[:N], K, mode='full')[:N])