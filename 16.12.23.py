import numpy as np
from functions import *
import scipy
from matplotlib import pyplot as plt 



N = 10**2
a = 0
b = 0.01
h = (b - a) / N
x_i_arr = np.zeros((N, N))

f_0 = 300
mu = 8 * 10**5
beta = 3 * 10**3


t_arr = np.linspace(a, b, N)
x_exact = np.cos(2 * np.pi * f_0 * t_arr + mu * t_arr**2)
K = np.exp(- beta * t_arr)
f_exact = scipy.signal.convolve(x_exact, K, mode='full')
f_exact = f_exact[:len(t_arr)]
freq_arr = scipy.fft.fftfreq(len(t_arr), b - a)

noise = np.random.normal(0.0, 1e-01, N)
f_noised = f_exact + noise

M_arr = np.linspace(freq_arr[1], freq_arr[-1], len(freq_arr))
M_arr = M_arr**2
M_arr = np.concatenate((M_arr[len(M_arr) // 2:], M_arr[:len(M_arr) // 2]))

alphas = np.logspace(-6, -1, N)

for i in range(N):
    x_i_arr[i] = Tikhonov_solver(f_noised, K, alphas[i], M_arr) 
    
x_averaged = np.zeros_like(x_exact)

for i in range(N):
    x_averaged[i] = np.sum(x_i_arr[:, i]) / N

plt.figure(dpi=500)
plt.plot(t_arr, x_exact, color='black')
plt.plot(t_arr[10:], x_averaged[10:], "--", color='red')
plt.legend(["exact", "averaged"])
plt.show()