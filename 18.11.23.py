import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
data = np.load("data_v9.npy")
N = data.shape[0]

np.random.seed(0)

a, b = min(data), max(data) # границы отрезка

def get_coefs(x, n=30):
    c, d = np.zeros((n, )), np.zeros((n, ))
    for i in range(n):
        c[i] = np.sum(np.cos(i*np.pi*x / 0.5))
        d[i] = np.sum(np.sin(i*np.pi*x / 0.5))
    return c/len(x), d/len(x)

x = np.linspace(a, b, 10**4)
c, d = get_coefs(data)
f = np.zeros_like(x)

for i in range(len(c)):
    f += (1 / 0.5) * (c[i]*np.cos(i*np.pi*x / 0.5) + d[i]*np.sin(i*np.pi*x / 0.5))

f -= c[0] 


fig_1 = plt.figure()

plt.plot(x, f)
plt.hist(data, bins=200, density=True)
plt.xlabel(r'$x$')
plt.ylabel(r'$f$')
plt.title("Аппроксимация по полной выборке")
fig_1.set_dpi(300.0)
plt.show()


rnd_small_data = np.random.choice(data, size=10**3, replace=False)

a, b = min(rnd_small_data), max(rnd_small_data)

x = np.linspace(a, b, 10**4)
c, d = get_coefs(rnd_small_data)
rnd_small_f = np.zeros_like(x)

for i in range(len(c)):
    rnd_small_f += (1 / 0.5) * (c[i]*np.cos(i*np.pi*x / 0.5) + d[i]*np.sin(i*np.pi*x / 0.5))

rnd_small_f -= c[0] 

fig_2 = plt.figure()
plt.plot(x, rnd_small_f)
plt.hist(rnd_small_data, bins=200, density=True)
plt.xlabel(r'$x$')
plt.ylabel(r'$f$')
plt.title("Аппроксимация по малой выборке без регуляризации")
fig_2.set_dpi(300.0)
plt.show()



def get_coefs_reg(x, n=30, alpha=1e-2):
    c, d = np.zeros((n, )), np.zeros((n, ))
    for i in range(n):
        c[i] = np.sum(np.cos(i*np.pi*x / 0.5)) / (1 + alpha * i**2)
        d[i] = np.sum(np.sin(i*np.pi*x / 0.5)) / (1 + alpha * i**2)
    return c/len(x), d/len(x)

c_reg, d_reg = get_coefs_reg(rnd_small_data)
reg_f = np.zeros_like(x)
for i in range(len(c_reg)):
    reg_f += (1 / 0.5) * (c_reg[i]*np.cos(i*np.pi*x / 0.5) + d_reg[i]*np.sin(i*np.pi*x / 0.5))
    
reg_f -= c_reg[0]

fig_3 = plt.figure()
plt.plot(x, reg_f)
plt.hist(rnd_small_data, bins=200, density=True)
plt.xlabel(r'$x$')
plt.ylabel(r'$f$')
plt.title("Аппроксимация по малой выборке с регуляризацией")
fig_3.set_dpi(300.0)
plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
