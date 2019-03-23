import math
import random
import numpy as np

def fourier(o, P, v, phi):
    y = []
    for i in range(numfeat - len(y)):
        arg = 0
        for j in range(len(o)):
            arg += P[i][j] * o[j]
        arg /= v
        arg += phi[i]
        y.append(math.sin(arg))
    return np.array(y)

def radial_basis_functions(o, p):
    rbf = []
    P = p[0]
    sigma = p[2]
    for i in range(numfeat):
        arg = 0
        for j in range(len(o)):
            arg += np.square(P[i][j] - o[j]) / sigma[j]
        rbf.append(np.exp(-arg))
    return np.array(rbf)

def polynomial(o, p):
    poly = []
    P = p[0]
    for i in range(numfeat):
        arg = 1
        for j in range(len(o)):
            pij = P[i][j]
            if pij != 0:
                arg *= (o[j] ** pij) / np.exp(pij)
        poly.append(arg)
    return np.array(poly)
