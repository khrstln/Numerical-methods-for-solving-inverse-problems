# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 13:43:11 2023

@author: khrstln
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

N = 10**2 # Количество точек, используемых при нахождении численного  решения с помощью квадратурных формул
right_border = 1 # Правая граница рассматриваемого отрезка
x = np.linspace(0, right_border, num=N)
h = right_border/N # Шаг при интегрировании с помощью квадратурных формулы

f = np.array([1.0 for _ in range(N)]) # Значения правой части уравнения

K = np.zeros((N, N)) # Двумерный массив, содержащий значения ядра уравнения
for i in range(N):
    for j in range(i+1):
        K[i][j] = np.exp(x[i])*np.exp(-x[j])
        
# Функция, возращающая значения решения с помощью квадратурных формул
def solver_1(x, K, f, h, y_0=1.0, a=0.0):
    y = np.array([y_0 for _ in range(len(x))])
    for i in range(1, len(x)):
        SUM = np.sum(np.dot(K[i][1:i], y[1:i]))
        y[i] = (f[i] + h/2 * K[i][0]*y_0 + h * SUM) / (1 - h/2 * K[i][i])
    return y


num_y_1 = solver_1(x, K, f, h)

# Функция, возвращающая значения аналитического решения уравнения
def analytics_f(x):
    return 0.5*(1 + np.exp(2*x))

exact_y = np.array([analytics_f(i) for i in x])


plt.figure(1)
plt.plot(x, num_y_1, color="r", linewidth=7, alpha=0.75)
plt.plot(x, exact_y, "black")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['Numerical solution 1','Exact solution'], loc=2)
plt.title("Solution 1")

M = 100 # Количество учитываемых слагаемых в ряду
phi_arr = np.array([np.array([1.0 for _ in range(len(x))]) for _ in range(M)]) # Инициализируем массив, содержащий значения функций phi_m в различных точках рассматриваемого отрезка
phi_arr[0] = f
for i in range(1, M):
    for j in range(len(x)):
        SUM = np.sum(np.dot(K[j][1:j], phi_arr[i-1][1:j]))
        phi_arr[i][j] = (h/2 * (K[i][0]*phi_arr[i-1][0] + K[i][len(x) - 1]*phi_arr[i-1][len(x) - 1]) + h * SUM) # Заполняем массив соответствующими значениями, используя квадратурные формулы
        
# Функция, возвращающая значения решения уравнения с помощью итерационного метода
def solver_2(x, K, f, h, M=M): 
    y = np.array([0.0 for _ in range(len(x))])
    for i in range(M):
        y = y + phi_arr[i]
    return y

num_y_2 = solver_2(x, K, f, h)

plt.figure(2)
plt.plot(x, num_y_2, color="cyan", linewidth=8, alpha=0.75)
plt.plot(x, exact_y, "black")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(['Numerical solution 2','Exact solution'], loc=2)
plt.title("Solution 2")

error_arr = np.array([np.sum((solver_2(x, K, f, h, M=i) - exact_y)**2) / len(exact_y) for i in range(M)])

plt.figure(3)
plt.plot([i for i in range(1, M + 1)], error_arr, color="black")
plt.yscale('log')
# plt.xscale('log')
plt.xlabel("Количество учитываемых слагаемых")
plt.ylabel("Mean square error")
plt.title("Error for solution 2")


plt.show



# %%
