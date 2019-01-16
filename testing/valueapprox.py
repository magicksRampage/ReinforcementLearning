import math
import random
import numpy as np
import matplotlib.pyplot as plt

def getphi(I):
    phi = []
    for i in range(I):
        phi.append(random.random() * 2 * math.pi - math.pi)
    return phi


def getP(I,J):
    P = []
    for i in range(I):
        Pi = []
        for j in range(J):
            Pi.append(random.gauss(0,1))
        P.append(Pi)
    return P
    

def fourier(o, P, v, phi):
    #return np.array(o)
    y = []
    for i in range(numfeat):
        arg = 0
        for j in range(len(o)):
            arg += P[i][j] * o[j]
        arg /= v
        arg += phi[i]
        y.append(math.sin(arg))
    return np.array(y)

def polynomial(x, deg):
    feat = np.zeros(deg+1)
    for i in range(deg+1):
        feat[i] = math.pow(x, i)
    return feat

def binning(x, max_x, min_x, numfeat):
    feat = np.zeros(numfeat)
    step = (max_x - min_x) / numfeat
    index = int(math.floor((x - min_x) / step))
    index= max(min(index, numfeat-1), 0)
    feat[index] = 1
    return feat

def getNoisySample(x):
    return random.gauss(math.sin(x), 0.1)


numfeat = 20
numobs = 1
P = getP(numfeat,numobs)
v = 0.14
phi = getphi(numfeat)
omega = []
for i in range(numfeat):
    omega.append(random.gauss(0,1))
    #omega.append(0.)
omega = np.array(omega)
print(P)
samples = np.array([])
x = np.arange(-np.pi, np.pi, 0.01)
y = np.sin(x)
i = 0
lrate = 1
samples = []
dOmega = [1]
mse = []
mse_old = 1
mse_new = 10
while True: #np.abs(mse_old - mse_new) > 0.0001:
    mse_old = mse_new
    i += 1
    lrate = 100 /(100+ i)
    s = random.random() * 2 * np.pi - np.pi
    samples = np.append(samples, s)
    lgoal = np.sin(s)
    #lgoal = getNoisySample(s)
    feat = fourier([s], P, v, phi)
    #feat = np.append(feat, 1)
    #feat = binning(s, np.pi, -np.pi, numfeat)
    #feat = polynomial(s, numfeat)
    pred =  feat * omega
    dOmega = (lgoal - pred) * feat
    omega += lrate * dOmega
    #print(dOmega)
    approx = np.zeros(len(x))
    if np.mod(i, 10000) == 0:
        for ns in range(len(x)):
            feat = fourier([x[ns]], P, v, phi)
            #feat = np.append(feat, 1)
            #feat = binning(x[ns], np.pi, -np.pi, numfeat)
            #feat = polynomial(x[ns], numfeat)
            approx[ns] = np.dot(feat,omega)
        mse_new = np.sum(np.square(approx - y)) / len(x)
        mse.append(mse_new)
        print(omega)
        plt.scatter(x, y)
        plt.scatter(x, approx)
        plt.show()
print(i)
plt.semilogy(range(len(mse)), mse)
plt.show()
