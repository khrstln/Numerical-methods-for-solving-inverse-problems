# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 00:26:52 2023

@author: khrstln
"""
import numpy as np
from scipy.fft import fftfreq, fft, ifft
from matplotlib import pyplot as plt
import scipy

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


def adaptrls(x, d, p, Lambda=1, alpha=1e05):
    N = len(x)

    w = np.zeros(p)  
    R_k_inv = alpha * np.eye(p)
    e = np.zeros(N)  
   
    for k in range(p, N):
        x_k = np.flip(x[max(0, k-p+1):k+1], axis=0)
       
        y_k = np.dot(w, x_k)
       
        e[k] = d[k] - y_k
        
        K = np.dot(R_k_inv, x_k) / (Lambda + np.dot(np.dot(x_k, R_k_inv), x_k))
       
        w += K * e[k]

        R_k_inv = (R_k_inv - np.outer(K, np.dot(x_k, R_k_inv))) / Lambda
   
    return w, e


def draw_dots(x: np.array, f: np.array, x_lbl=None, y_lbl=None, title=None, color=None, size=None):
    plt.scatter(x, f, s=size, color=color)
    if x_lbl is not None:
        plt.xlabel(x_lbl)
    if y_lbl is not None:
        plt.ylabel(y_lbl)
    if title is not None:
        plt.title(title)
    # plt.show()
    
def draw_func(x: np.array, f: np.array, x_lbl=None, y_lbl=None, title=None, color=None, linewidth=None):
    plt.plot(x, f, color=color, linewidth=linewidth)
    if x_lbl is not None:
        plt.xlabel(x_lbl)
    if y_lbl is not None:
        plt.ylabel(y_lbl)
    if title is not None:
        plt.title(title)
    # plt.show()
    
def Tikhonov_solver(f_noise: np.array, K: np.array, alpha: float, M: np.array):
    return scipy.fft.ifft(scipy.fft.fft(f_noise) * np.conj(scipy.fft.fft(K)) / (scipy.fft.fft(K) * np.conj(scipy.fft.fft(K)) + alpha * M))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
