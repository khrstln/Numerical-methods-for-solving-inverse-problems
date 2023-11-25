# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:26:52 2023

@author: khrstln
"""
import numpy as np
from scipy.fft import fftfreq, fft, ifft

def der_two_points(func_arr: np.array, dt: float):
    return (np.roll(func_arr, -1) - np.roll(func_arr, 1)) / (2*dt)

def M(omega_arr: np.array, omega_0: float):
    M_arr = np.ones_like(omega_arr)
    for i in range(len(omega_arr)):
        if abs(omega_arr[i]) > omega_0:
            M_arr[i] = 0.
    return M_arr

def der_Fourier(func_arr: np.array, t_arr: np.array, dt: float, omega_0: float):
    F_omega = fft(func_arr)
    omega = 2 * np.pi * fftfreq(len(func_arr), dt)
    M_arr = M(omega, omega_0)
    return omega, np.real(ifft(1j * omega * F_omega  * M_arr))
    
