import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-2
t = np.arange(-10, 10 + steps , steps)


def s(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
        else:
            y[i] = 0
    return y


def h1(t):
    y = np.zeros(t.shape)
    y = (s(t) - s(t - 3)) * np.exp(-t*2)
    return y

def h2(t):
    y = np.zeros(t.shape)
    y = s(t - 2) - s(t - 6)
    return y

def h3(t):
    y = np.zeros(t.shape)
    ω0 = 0.25*np.pi*2
    y = np.cos(ω0*t)*s(t)
    return y



def y1(t):
    y = np.zeros(t.shape)
    y = 0.5*((-np.exp(-2*t)+1)*s(t) - (-np.exp(-2*(t-3))+1)*s(t-3))
    return y

def y2(t):
    y = np.zeros(t.shape)
    y = (t-2)*s(t-2) - (t-6)*s(t-6)
    return y

def y3(t):
    y = np.zeros(t.shape)
    ω0 = 0.25*np.pi*2
    y =  (1/ω0)*np.sin(ω0*t)*s(t)
    return y




q1 = sig.convolve(h1(t),s(t)) * steps

q2 = sig.convolve(h2(t),s(t)) * steps

q3 = sig.convolve(h3(t),s(t)) * steps

tconv = np.arange(2*t[0], 2*t[len(t)-1]+steps, steps)

plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t, h1(t))
plt.title('Transfer Functions')
plt.ylabel('h1')
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(t, h2(t))
plt.ylabel('h2')
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(t, h3(t))
plt.ylabel('h3')
plt.grid(True)


plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(tconv, q1)
plt.title('Step Responses')
plt.ylabel('h1')
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(tconv, q2)
plt.ylabel('h2')
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(tconv, q3)
plt.ylabel('h3')
plt.grid(True)

plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(t, y1(t))
plt.title('Hand Calculaion Step Responses')
plt.ylabel('h1')
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(t, y2(t))
plt.ylabel('h2')
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(t, y3(t))
plt.ylabel('h3')
plt.grid(True)
