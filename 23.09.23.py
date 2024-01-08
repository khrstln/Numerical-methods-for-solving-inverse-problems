import numpy as np
import scipy
from matplotlib import pyplot as plt
from reg_Tikhonov import Tikhonov_solver
a = 0
b = 0.01
N = 10**4
h = (b - a) / N
t_arr = np.linspace(a, b, N)
x_arr = np.cos(2 * np.pi * 300 * t_arr + 8 * 10**5 * t_arr**2)
K_arr = np.exp(- 3 * 10**3 * t_arr)
freq_arr = scipy.fft.fftfreq(len(t_arr), b - a)

# Задание № 1

f_arr = scipy.signal.convolve(x_arr, K_arr, mode='full')
f_arr = f_arr[:len(t_arr)]
plt.plot(t_arr, f_arr)
plt.xlabel("t")
plt.plot("f(t)")

plt.show()

# Задание № 2

sigma_1 = 0.5
sigma_2 = 0.005
noise_arr_1 = np.random.normal(0, sigma_1, size=N)
noise_arr_2 = np.random.normal(0, sigma_2, size=N)

f_noise_arr_1 = f_arr + noise_arr_1
f_noise_arr_2 = f_arr + noise_arr_2

# Задание № 3

M_arr = np.linspace(freq_arr[1], freq_arr[-1], len(freq_arr))
M_arr = M_arr**2
M_arr = np.concatenate((M_arr[len(M_arr) // 2:], M_arr[:len(M_arr) // 2]))
alpha = 1
x_alpha_1 = Tikhonov_solver(f_noise_arr_1, K_arr, alpha, M_arr)
x_alpha_2 = Tikhonov_solver(f_noise_arr_2, K_arr, alpha, M_arr)

# x_alpha_1 = np.real(x_alpha_1)
# x_alpha_2 = np.real(x_alpha_2)

alpha_arr = np.logspace(10, 12, 5*10**2)

error_arr = [[], []]

for alpha in alpha_arr:
    error_1 = np.linalg.norm(Tikhonov_solver(f_noise_arr_1, K_arr, alpha, M_arr) - x_arr) / np.linalg.norm(x_arr)
    error_2 = np.linalg.norm(Tikhonov_solver(f_noise_arr_2, K_arr, alpha, M_arr) - x_arr) / np.linalg.norm(x_arr)
    error_arr[0].append(error_1)
    error_arr[1].append(error_2)
    
# print(np.linalg.norm(np.array([0, 0]) - np.array([1, 1])))

plt.semilogx(alpha_arr, error_arr[0], color='black')
plt.semilogx(alpha_arr, error_arr[1], color="red", linewidth=8, alpha=0.5)
# plt.scatter(alpha_arr[error_arr[0].index(min(error_arr[0]))], min(error_arr[0]))
plt.xlabel("$\\alpha$")
plt.ylabel("error")
plt.legend([f"$\sigma_1^2 = {sigma_1}$", f"$\sigma_2^2 = {sigma_2}$"])
plt.show()

plt.plot(t_arr[1:], x_arr[1:], color="red", linewidth=5, alpha=0.5)
# plt.plot(t_arr[1:], x_alpha_1[1:])
plt.plot(t_arr[1:], x_alpha_2[1:])
plt.xlabel("t")
plt.ylabel("$x_{\\alpha} (t)$")
plt.legend(["$\sigma^2 = 0$", f"$\sigma_2^2 = {sigma_2}$"])

plt.show()

# Задание №4

num_of_exp = 10**2
error_arr = [[[], []] for i in range(num_of_exp)]

for i in range(num_of_exp):
    print(i)
    noise_arr_1 = np.random.normal(0, sigma_1, size=N)
    noise_arr_2 = np.random.normal(0, sigma_2, size=N)
    f_noise_arr_1 =  f_arr + noise_arr_1
    f_noise_arr_2 =  f_arr + noise_arr_2
    for alpha in alpha_arr:
        x_alpha_1 = Tikhonov_solver(f_noise_arr_1, K_arr, alpha, M_arr)
        x_alpha_2 = Tikhonov_solver(f_noise_arr_2, K_arr, alpha, M_arr)
        error_1 = np.linalg.norm(x_alpha_1 - x_arr)
        error_2 = np.linalg.norm(x_alpha_2 - x_arr)
        error_arr[i][0].append(error_1)
        error_arr[i][1].append(error_2)
alpha_opt = {sigma_1: [], sigma_2: []}

for i in range(num_of_exp):   
    alpha_opt[sigma_1] += [alpha_arr[np.where(error_arr[i][0] == min(error_arr[i][0]))][0]]
    alpha_opt[sigma_2] += [alpha_arr[np.where(error_arr[i][1] == min(error_arr[i][1]))][0]]

plt.hist(np.log(alpha_opt[sigma_1]), bins=None)
plt.xlabel("$\\log(\\alpha_{opt})$")
plt.ylabel("frequency")
plt.title(f"$\\sigma_1 = {sigma_1}$")
plt.show()

plt.hist(np.log(alpha_opt[sigma_2]), bins=None)
plt.xlabel("$\\log(\\alpha_{opt})$")
plt.ylabel("frequency")
plt.title(f"$\\sigma_2 = {sigma_2}$")
plt.show()
 
    






























