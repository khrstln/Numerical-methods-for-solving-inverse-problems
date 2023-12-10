# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:46:37 2023

@author: khrstln
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt
from iter_Fridman import Get_next_x
a = 0
b = 0.01
N = 2 * 10**2
h = (b - a) / N
x_arr = np.array([np.cos(2 * np.pi * 300 * (i * h) + 8 * 10**5 *(i * h)**2) for i in range(N)]) # Точное решение
x_norm = np.linalg.norm(x_arr)**2
K_arr = np.array([np.exp(- 3 * 10**3 * (i * h)) for i in range(N)]) # Ядро
t_arr = np.linspace(a, b, N)
freq_arr = scipy.fft.fftfreq(len(t_arr), b - a)
f_arr = np.convolve(x_arr, K_arr, mode='full')

# f_check = np.real(np.fft.ifft(np.fft.fft(x_arr) * np.fft.fft(K_arr)))
# f_check1 = np.real(np.fft.ifft(np.fft.fft(x_arr, 2*256) * np.fft.fft(K_arr, 2*256)))
# plt.plot(f_arr, '-k')
# plt.plot(f_check1,'--r')
# plt.plot(f_check,':b')
# plt.show()


sigma = 0.25
noise_arr = np.random.normal(0, sigma, size=2*N - 1)
f_noise_arr = f_arr + noise_arr

# plt.plot(t_arr, f_arr, color="g")
# plt.xlabel("t")
# plt.title("Exact f(t)")
# plt.plot("f(t)")
# plt.show()

x_iter_1 = np.array([0.0 for i in range(N)])
x_iter_2 = np.array([0.0 for i in range(N)])

lambda_max = np.max(np.abs(np.linalg.eigvals(scipy.linalg.circulant(K_arr)))) # Находим максимальное собственное значения матрицы левой части 

temp = scipy.linalg.circulant(K_arr)

alpha_num = 4 # Число значения альфа 
# alpha_arr = [(2 * lambda_max) / i for i in range(1, alpha_num + 1)]
alpha_arr = np.linspace(0.01, (2 / lambda_max), alpha_num)
alpha_arr = np.array([0.005, 0.0075, 0.01, 0.02])


errors = [[] for _ in range(alpha_num)] # Список значений ошибки
MAX_ITER = 400
eps = 10**(-9)
norma = 1

x_iter_2 = Get_next_x(f_noise_arr, K_arr, x_iter_1, alpha_arr[0], N)

i = 0
for alpha in alpha_arr:
    j = 0
    while j < MAX_ITER and norma > eps:
        x_iter_2 = Get_next_x(f_noise_arr, K_arr, x_iter_1, alpha, N)
        norma = np.linalg.norm(x_iter_2 - x_arr)**2 / x_norm
        errors[i].append(norma)
        x_iter_1  = np.copy(x_iter_2)
        j += 1
    x_iter_1 = np.array([0.0 for i in range(N)])
    x_iter_2 = np.array([0.0 for i in range(N)])
    x_iter_2 = Get_next_x(f_noise_arr, K_arr, x_iter_1, alpha_arr[0], N)
    plt.semilogy([l for l in range(1, j + 1)], errors[i])
    plt.xlabel("m")
    plt.ylabel("error")
    i += 1

plt.ylim(0, 3e-1)
plt.legend([f"alpha = {alpha_arr[i]}" for i in range(len(alpha_arr))])
plt.show()

# plt.plot(t_arr, x_iter_2, color="red")
# plt.plot(t_arr, x_arr)
# plt.show()
 