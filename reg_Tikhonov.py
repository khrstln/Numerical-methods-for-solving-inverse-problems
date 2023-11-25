import numpy as np
import scipy

def Tikhonov_solver(f_noise: np.array, K: np.array, alpha: float, M: np.array):
    return scipy.fft.ifft(scipy.fft.fft(f_noise) * np.conj(scipy.fft.fft(K)) / (scipy.fft.fft(K) * np.conj(scipy.fft.fft(K)) + alpha * M))