# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 09:38:37 2023

@author: khrstln
"""

#Работа №2: 

import numpy as np
import matplotlib.pyplot as plt

lambda_ = 1.0
beta = 1.0
a = 0.0
b = 2 * np.pi
N = 50
h = (b - a) / N

x_arr = np.linspace(a, b, N) # Массив, содержащий значения переменной
K_func = lambda x, t: np.cos(beta * x) # Функция, вычисляющая значение ядра в точке (x, t)
K_arr = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        K_arr[i][j] = K_func(x_arr[i], x_arr[j])
        
f_arr = np.ones(N) # Массив, содержащий значения правой части уравнения
sigma = 1e-01 # среднеквадратическое отклонение для шума
f_noised = f_arr + np.random.normal(0.0, sigma, N)


y_exact = 1 + 2 * np.pi * np.cos(x_arr) # Точное решение интегрального уравнения при a = 0, b = 2*PI

# Функция, возвращающая решение с помощью квадратурной формулы
def Solver_quadr(x, K, f, a, b, h):
    n = len(x)
    wt=1/2
    wj=1
    A = np.zeros((n, n))
    for i in range(n):
        A[i][0]= -h * wt * K(x[i], x[0])
        for j in range(1, n - 1):
            A[i][j] = -h * wj * K(x[i], x[j])
        A[i][n - 1] = -h * wt * K(x[i], x[n - 1])
        A[i][i] = A[i][i] + 1
        B = np.zeros((n,1))
        for j in range(n):
            B[j][0] = f[j]
    y = np.linalg.solve(A, B)
    return y

y_numerical = Solver_quadr(x_arr, K_func, f_arr, a, b, h)

plt.plot(x_arr, y_exact, color='black')
plt.scatter(x_arr, y_numerical, s=12, c=[(0.8, 0, 0)])
plt.xlabel("$x$")
plt.ylabel("$y(x)$")
plt.title("Квадратурные формулы, правая часть без шума")
plt.show()

y_numerical = Solver_quadr(x_arr, K_func, f_noised, a, b, h)

plt.plot(x_arr, y_exact, color='green')
plt.scatter(x_arr, y_numerical, s=12, c=[(0, 0, 0.8)])
plt.xlabel("$x$")
plt.ylabel("$y(x)$")
plt.title(f"Квадратурные формулы, правая часть зашумлена \n $\sigma = ${sigma}")
plt.show()




