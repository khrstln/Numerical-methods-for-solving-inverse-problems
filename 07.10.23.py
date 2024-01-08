import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functions import Tikhonov_solver
import scipy

np.random.seed(0)

# Lab 5

a = 0
b = 0.01
N = 10**4
h = (b - a) / N
t_arr = np.linspace(a, b, N)
f_0 = 300
mu = 8 * 10**5
beta = 3 * 10**3

x = lambda t: np.cos(2 * np.pi * f_0 * t_arr + mu * t**2)
K = lambda t, beta: np.exp(- beta * t)

x_exact = x(t_arr)
K_exact = K(t_arr, beta)

# Task 1

f_exact = scipy.signal.convolve(x_exact, K_exact, mode='full')
f_exact = f_exact[:len(t_arr)]

offset = beta * 1e-01
beta_offseted = beta + offset
K_offseted = K(t_arr, beta_offseted)


f_offseted = scipy.signal.convolve(x_exact, K_offseted, mode='full')
f_offseted = f_offseted[:len(t_arr)]
plt.figure(dpi=500)
plt.plot(t_arr, f_exact, color="black")
plt.plot(t_arr, f_offseted, "--", color="red")
plt.xlabel("t")
plt.legend(["$f_{exact}(t)$", "$f_{offseted}(t)$"])
# plt.plot("f(t)")

plt.show()

# Task 2

noise = np.random.normal(0.0, 1e-02, N)

f_offseted_noised = f_offseted + noise
alpha = 1.0

freq_arr = scipy.fft.fftfreq(len(t_arr), b - a)
M_arr = np.linspace(freq_arr[1], freq_arr[-1], len(freq_arr))
M_arr = M_arr**2
M_arr = np.concatenate((M_arr[len(M_arr) // 2:], M_arr[:len(M_arr) // 2]))

x_alpha = Tikhonov_solver(f_offseted_noised, K_exact, alpha, M_arr)

plt.figure(dpi=500)
plt.plot(t_arr[1:], x_exact[1:], color="black")
plt.plot(t_arr[1:], x_alpha[1:], "--", color="red")
plt.xlabel("t")
plt.legend(["$x_{exact}(t)$", "$x_{\\alpha}(t)$"])
plt.title(f"$\\alpha = {alpha}$")

plt.show()

# Task 3

n = 10**2

beta_offseted_arr = np.linspace(beta, beta * 5, n)
errors = np.zeros((1, n))
errors = errors[0]

for i, beta_offseted in enumerate(beta_offseted_arr):
    K_offseted = K(t_arr, beta_offseted)
    
    f_offseted = scipy.signal.convolve(x_exact, K_offseted, mode='full')
    f_offseted = f_offseted[:len(t_arr)]
    
    # noise = np.random.normal(0.0, 1e-02, N)
    
    f_offseted_noised = f_offseted + noise
    
    x_alpha = Tikhonov_solver(f_offseted_noised, K_exact, alpha, M_arr)
    
    errors[i] = np.linalg.norm(x_exact[:-1]-x_alpha[1:])**2 / np.linalg.norm(x_exact[:-1])**2
    
print(x_alpha[1:])
plt.figure(dpi=500)
plt.plot((beta_offseted_arr - beta) / beta, errors)
plt.xlabel(r"$\delta \beta / \beta$")
plt.ylabel(r"MSE")
plt.title(r"$\frac{|x_{exact} - x_{\alpha}|^2}{|x_{exact}|^2}$")
plt.show()
    






















