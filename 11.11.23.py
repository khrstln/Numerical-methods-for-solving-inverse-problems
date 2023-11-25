# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 23:57:39 2023

@author: khrstln
"""
import numpy as np
from scipy.fft import fftfreq, fft, ifft
from matplotlib import pyplot as plt
from functions import *
 
N = 10**3 
a = 0.0
b = 0.2
dt = (b - a) / (N + 1) 
t_arr = np.linspace(a, b, N + 1)
f_1_arr = np.cos(20*np.pi*t_arr)**2


# Задание №1
der_f_1_analitical_arr = -20*np.pi*np.sin(40*np.pi*t_arr)


# Задание №2.1 

plt.plot(t_arr, der_f_1_analitical_arr, color='black')
plt.plot(t_arr, der_two_points(f_1_arr, dt), color="red", linewidth=7, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical", "numerical two points"], loc=4)
plt.title("Функция 1, без шума, по 2 точкам")
plt.show()

omega_0 = 2 * np.pi * 20
omega_arr, der_f_Fourier_arr = der_Fourier(f_1_arr, t_arr, dt, omega_0)

plt.plot(t_arr, der_f_1_analitical_arr, color='black')
plt.plot(t_arr, der_f_Fourier_arr, color="#00CC66", linewidth=7, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical", "numerical Fourier"], loc=4)
plt.title("Функция 1, без шума, Фурье")
plt.show()

# Задание №2.2

sigma_1 = 0.05
sigma_2 = 0.5

f_1_noised_arr_1 = f_1_arr + np.random.normal(0, sigma_1, size=N+1)
f_1_noised_arr_2 = f_1_arr + np.random.normal(0, sigma_2, size=N+1)

# По двум точкам
plt.plot(t_arr, der_f_1_analitical_arr, color='black')
plt.plot(t_arr, der_two_points(f_1_noised_arr_1, dt), color="#8A2BE2", linewidth=1, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical", "numerical two points"], loc=4)
plt.title(f"Функция 1, $\\sigma = {sigma_1}$, по 2 точкам")
plt.show() 

plt.plot(t_arr, der_f_1_analitical_arr, color='black')
plt.plot(t_arr, der_two_points(f_1_noised_arr_2, dt), color="#4169E1", linewidth=1, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical", "numerical two points"], loc=4)
plt.title(f"Функция 1, $\\sigma = {sigma_2}$, по 2 точкам")
plt.show()

# Фурье

omega_arr, der_f_Fourier_arr = der_Fourier(f_1_noised_arr_1, t_arr, dt, omega_0)

plt.plot(t_arr, der_f_1_analitical_arr, color='black')
plt.plot(t_arr, der_f_Fourier_arr, color="#1E90FF", linewidth=7, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical", "numerical Fourier"], loc=4)
plt.title(f"Функция 1, $\\sigma = {sigma_1}$, Фурье")
plt.show()

omega_arr, der_f_Fourier_arr = der_Fourier(f_1_noised_arr_2, t_arr, dt, omega_0)

plt.plot(t_arr, der_f_1_analitical_arr, color='black')
plt.plot(t_arr, der_f_Fourier_arr, color="#90EE90", linewidth=7, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical", "numerical Fourier"], loc=4)
plt.title(f"Функция 1, $\\sigma = {sigma_2}$, Фурье")
plt.show()


# Задание №3

a = -5
b = 5
dt = (b - a) / (N + 1) 
t_arr = np.linspace(a, b, N + 1)
f_2_arr = t_arr**3 + 5*t_arr
der_f_2_analitical_arr = 3 * t_arr**2 + 5

f_2_noised_arr = f_2_arr + np.random.normal(0, sigma_1, size=N+1)

plt.plot(t_arr, der_f_2_analitical_arr, color='black')
plt.plot(t_arr[1:len(t_arr) - 1], der_two_points(f_2_noised_arr, dt)[1:len(t_arr) - 1], color="#FF8C00", linewidth=1, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical", "numerical two points"], loc=0)
plt.title(f"Функция 2, $\\sigma = {sigma_1}$, по 2 точкам")
plt.show() 

omega_0 = 2 * np.pi * 5
omega_arr, der_f_Fourier_arr = der_Fourier(f_2_noised_arr, t_arr, dt, omega_0)

plt.plot(t_arr, der_f_2_analitical_arr, color='black')
plt.plot(t_arr, der_f_Fourier_arr, color="#FF6347", linewidth=1, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical", "numerical two points"], loc=0)
plt.title(f"Функция 2, $\\sigma = {sigma_1}$, Фурье")
plt.show() 

c = np.polyfit(t_arr, f_2_noised_arr, deg=3)

plt.plot(t_arr,  der_f_2_analitical_arr, color='black')
plt.plot(t_arr, 3*t_arr**2*c[0] + 2*t_arr*c[1] + c[2], color="cyan", linewidth=5, alpha=0.5)
plt.xlabel("t")
plt.ylabel("derivatives")
plt.legend(["analitical: $3t^2 + 5$", "Least square method"])
plt.title(f"Функция 2, $\\sigma = {sigma_1}$, МНК")
plt.show()



























