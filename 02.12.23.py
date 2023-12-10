# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:59:38 2023

@author: khrstln
"""

import numpy as np
import random
from functions import draw_dots, draw_func
from sklearn.linear_model import RANSACRegressor
from matplotlib import pyplot as plt
# np.random.seed(0)

a, b = 0, 1
N = 10**2
h = (b - a) / N
t_arr = np.linspace(a, b, N)

a_1 = 1
a_0 = 0.1
sigma = 10**(-2)
x_arr = a_1 * t_arr + a_0 + np.random.normal(0, sigma, N)


p_v = 0.3
for i in range(N):
    if np.random.rand() < p_v:
        x_arr[i] += 0.5 + np.random.rand()
        
draw_dots(t_arr, x_arr, x_lbl="t", y_lbl="x(t)", title="Зашумленные данные", color="black", size=10)

RANSAC = RANSACRegressor(stop_n_inliers=50, max_trials=5)
RANSAC.fit([[t_arr[i]] for i in range(N)], x_arr)
x_predict = RANSAC.predict([[t_arr[i]] for i in range(N)])

draw_func(t_arr, x_predict, x_lbl="t", y_lbl="x(t)", title="RANSAC predict", color="red", linewidth=2)

plt.show()

