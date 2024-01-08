# ЛР №9
from functions import *
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
np.random.seed(0)

N = 10**3
p = 16

h = np.random.random(p)
x = np.random.random(N)

sigma = 1e-4


d = signal.lfilter(h, 1, x) + np.random.normal(0, sigma, len(x))

w, e = adaptrls(x, d, p, Lambda=1.0, alpha=1e05)


plt.scatter([k for k in range(len(e))], e, s=12)
plt.xlabel("Номер шага, $k$")
plt.ylabel("$e(k)$")
plt.title(f"Зависимость ошибки от номера шага \n $p=${p}, $\sigma=${sigma}, $\lambda=${1.0}")
plt.yscale("log")
plt.show()

# Задания №3, №5
# Исследовать сходимость алгоритма в условиях различных значений sigma, p и Lambda

def test(N=N, sigma=1e-04, p=16, Lambda=0.95):
    h = np.random.random(p)
    x = np.random.random(N)
    d = signal.lfilter(h, 1, x) + np.random.normal(0, sigma, len(x))

    w, e = adaptrls(x, d, p, Lambda=0.95, alpha=1e05)
    plt.scatter([k for k in range(len(e))], e, s=12)
    plt.xlabel("Номер шага, $k$")
    plt.ylabel("$e(k)$")
    plt.title(f"Зависимость ошибки от номера шага \n $p=${p}, $\sigma=${sigma}, $\lambda=${Lambda}")
    plt.yscale("log")
    plt.show()
    
    return w, e

# Исследуем зависимость от sigma
w, e = test(sigma=1e-03)
w, e = test(sigma=1e-02)
w, e = test(sigma=1e-01)
w, e = test(sigma=1)

# Исследуем зависимость от Lambda
w, e = test(Lambda=0.2)
w, e = test(Lambda=0.4)
w, e = test(Lambda=0.7)
w, e = test(Lambda=2.0)

# Исследуем зависимость от p
w, e = test(p=10)
w, e = test(p=14)
w, e = test(p=18)
w, e = test(p=22)


# Задане №4

h1 = np.random.random(p)
h2 = np.random.random(p)
Lambda = 0.95

x = np.random.random(N)
d = [*signal.lfilter(h, 1, x[:len(x)//2]), *signal.lfilter(h, 1, x[len(x)//2:])] + np.random.normal(0, sigma, len(x))

w, e = adaptrls(x, d, p, Lambda=Lambda)
plt.scatter([k for k in range(len(e))], e, s=12)
plt.xlabel("Номер шага, $k$")
plt.ylabel("$e(k)$")
plt.title(f"Зависимость ошибки от номера шага \n $p=${p}, $\sigma=${sigma}, $\lambda=${1.0}")
plt.yscale("log")
plt.show()



    
    