import numpy as np
import matplotlib.pyplot as plt

steps = 1e-2
t = np.arange(0, 20 + steps , steps)

def r(t): 
    y = np.zeros(t.shape)
    for i in range(len(t)):
       if t[i] > 0:
           y[i] = t[i]
       else:
           y[i] = 0
    return y
    
def s(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
        else:
            y[i] = 0
    return y









def f1(t): 
    y = np.zeros(t.shape)
    y = s(t-2) - s(t-9)
    return y

def f2(t): 
    y = np.zeros(t.shape)
    y = s(t)*np.exp(-t)
    return y

def f3(t): 
    y = np.zeros(t.shape)
    y = r(t-2)*s(t-2) - r(t-2)*s(t-3) + r(4-t)*s(t-3) - r(4-t)*s(t-4)
    return y

y = f1(t)
x = f2(t)
z = f3(t)

def conv(f1,f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1E = np.append(f1, np.zeros((1,Nf2 - 1)))
    f2E = np.append(f2, np.zeros((1,Nf1 - 1)))
    result = np.zeros(f1E.shape)
    
    for i in range(Nf2+Nf1-2):
        result[i] = 0
        for j in range(Nf1):
            if(i-j+1>0):
                    result[i] = result[i] + (f1E[j]*f2E[i-j+1])
                
    return result
            
q1 = conv(y,x)*steps

q2 = conv(x,z)*steps

q3 = conv(y,z)*steps

tconv = np.arange(0, 40 + 2*steps, steps)

plt.figure(figsize = (10,7))
plt.subplot(3,1,1)
plt.plot(tconv, q3)
plt.title('f1 and f3')
#plt.ylabel('f1')
plt.grid(True)
#plt.subplot(3,1,2)
#plt.plot(t, x)
#plt.ylabel('f2')
#plt.grid(True)
#plt.subplot(3,1,3)
#plt.plot(t, z)
#plt.ylabel('f3')
#plt.grid(True)
plt.show()