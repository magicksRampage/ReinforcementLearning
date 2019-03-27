import math
import random
import numpy as np
import matplotlib.pyplot as plt

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
    for i in range(len(P)):
        arg = 0
        for j in range(len(o)):
            if P[i][j] != 0:
                arg += np.square(P[i][j] - o[j]) / sigma[j]
        rbf.append(np.exp(-arg))
    return np.array(rbf)

def polynomial(o, p):
    poly = []
    P = p[0]
    numfeat = P.shape[0]
    for i in range(numfeat):
        arg = 1
        for j in range(len(o)):
            pij = P[i][j]
            if pij != 0:
                arg *= o[j] ** pij
        poly.append(arg)
    return np.array(poly)

def tiling(o, p):
    tiles = []
    P = p[0]
    numfeat = P.shape[0]
    for feat in range(numfeat):
        obs1 = int(P[feat][0])
        obs2 = int(P[feat][3])
        if o[obs1] > P[feat][1] and o[obs1] < P[feat][2] and o[obs2] > P[feat][4] and o[obs2] < P[feat][5]:
            tiles.append(1)
        else:
            tiles.append(0)

    return np.array(tiles)

def getphi(I,env):
    phi = []
    #limits = env.observation_space.high
    #return np.array(limits) * 2
    return np.ones([I])
    for i in range(I):
        phi.append(random.random() * 2 * math.pi - math.pi)
    return phi


def getP(I,J,env):
    P = []
    limits = env.observation_space.high
    for i in range(I):
        Pi = []
        for j in range(J):
            if random.random() < 0.34:
                Pi.append(2 * random.random() - 1)
            else:
                Pi.append(0)
            #Pi.append(random.gauss(0,1))
        P.append(Pi)
    return P

def getP_polynomial(numobs, degree):
    numfeat = 0
    for i in range(degree):
        if i+1 == 2:
            numfeat += int(np.math.factorial(numobs) / (np.math.factorial(2) * np.math.factorial(numobs-2)))
        numfeat += numobs

    P = np.zeros([numfeat, numobs])
    current_row = 0
    for deg in range(degree):
        features_for_degree = numobs if deg+1 != 2 else np.math.factorial(numobs) / (np.math.factorial(degree) * np.math.factorial(numobs-degree))
        for obs in range(numobs):
            if deg+1 != 2:
                P[current_row][obs] = deg+1
                current_row += 1
            else:
                for obs2 in range(obs,numobs):
                    P[current_row][obs] += 1
                    P[current_row][obs2] += 1
                    current_row += 1

    return P

def getP_2dtiles(numfeat, numobs):
    P = np.zeros([numfeat, numobs])
    for feat in range(numfeat):
        for i in range(2):
            obs = random.choice(range(numobs))
            a = 2 * random.random() - 1
            b = 2 * random.random() - 1
            upper = max(a,b)
            lower = min(a,b)
            P[feat][i*3] = obs
            P[feat][i*3 + 1] = lower
            P[feat][i*3 + 2] = upper

    return P
        

def initialize_feature_parameters(num_features = 0, num_observations = 0, env = None, feature_type = "linear"):
    if feature_type == "linear":
        return
    elif feature_type == "polynomial":
        return [getP_polynomial(num_observations, num_features), 0, []]
    elif feature_type == "2dtiles":
        return [getP_2dtiles(num_features, num_observations), 0 , []]
    else:
        phi = getphi(num_observations, env = env)
        P = getP(num_features, num_observations, env = env)
        v = 1
        return [P, v, phi]

def gradient_update(stepsize = 0.1, value = [], feature_params = [], x = 0, y = 0):
    features = radial_basis_functions(o = [x], p = feature_params)
    y_est = np.dot(value, features)
    value += stepsize * features * (y - y_est)

    return value

def plot(value = [], feature_params = [], validation_set = []):
    y_est = []
    y_true = []
    for x in validation_set:
        y_true.append(np.sin(x))
        y_est.append(np.dot(value, radial_basis_functions(o = [x], p = feature_params)))

    plt.scatter(validation_set, y_true)
    plt.scatter(validation_set, y_est)
    plt.show()
    

def test_features():
    P = []
    num_feat = 50
    for i in range(num_feat):
        Pi = [2 * np.pi * random.random()]
        P.append(Pi)
    sigmas = np.ones([num_feat]) * 2 * np.pi

    params = np.array([P, 0, sigmas])
    
    value = []
    for i in range(num_feat):
        value.append(random.gauss(0,1))

    value = np.array(value)

    iterations = 100000

    grain = 50
    validation_set = np.array(range(grain)) * 2 * np.pi / grain

    for i in range(iterations):
        x = 2 * np.pi * random.random()
        y = random.gauss(np.sin(x), 0.001)

        value = gradient_update(stepsize = 500.0 / (i+10000), value = value, feature_params = params, x = x, y = y)
        print(value)
        if np.mod(i,1000) == 0:
            plot(value = value, feature_params = params, validation_set = validation_set)

#test_features()
