import math
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

def fourier(o, P, v, phi):
    """
    calculates fourier features

    :param o: vector of observations
    :param P: weight matrix for linear combination of observations
    :param v: wavelength parameter (float)
    :param phi: vector of offsets, one for each feature
    :return: vector of fourier features
    """
    y = []
    numfeat = len(phi)
    for i in range(numfeat):
        arg = 0
        for j in range(len(o)):
            arg += P[i][j] * o[j]
        arg /= v
        arg += phi[i]
        y.append(math.sin(arg))
    return np.array(y)

def radial_basis_functions(o, p):
    """
    calculates rbf features

    :param o: vector of observations
    :param p: feature parameters [P, 0, sigma]
    :return: vector of rbf features
    """
    rbf = []
    P = np.array(p[0])
    sigma = np.array(p[2])
    obs = np.array(o)
    args = np.square(P - obs) / sigma
    args = np.dot(args, np.ones(len(obs)))
    rbf = np.exp(-args)
    return rbf
    for i in range(len(P)):
        arg = 0
        for j in range(len(o)):
            arg += np.square(P[i][j] - o[j]) / sigma[j]
        rbf.append(np.exp(-arg))
    return np.array(rbf)

def polynomial(o, p):
    """
    calculates polynomial features

    :param o: vector of observations
    :param p: feature parameters [P, 0, 0]
    :return: vector of polynomial features
    """
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
    """
    calculates tile features constrained in 2 dimensions

    :param o: vector of observations
    :param p: feature parameters [P, 0, 0]
    :return: vector of tile features
    """
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
    """
    get random offsets for fourier features in (-pi, pi)

    :param I: number of features
    :param env: gym learning environment - deprecated
    :return: vector of random offsets
    """
    phi = []
    for i in range(I):
        phi.append(random.random() * 2 * math.pi - math.pi)
    return phi


def getP(I,J,env):
    """
    get random weight matrix for feature construction

    :param I: number of features
    :param J: number of observations
    :param env: gym learning environment
    :return: random weight matrix (:param I:,:param J:)
    """
    P = []
    limits = env.observation_space.high
    for i in range(I):
        Pi = []
        for j in range(J):
            if random.random() < 1:
                Pi.append(2 * random.random() - 1)
            else:
                Pi.append(0)
        P.append(Pi)
    return P

def getP_polynomial(numobs, degree):
    """
    get exponent matrix for polynomial features
    assumes weak coupling. only 2nd degree terms have more than exponent > 0

    :param numobs: number of observations
    :param degree: maximum degree of the polynomial
    :return: exponent matrix (number of features, :param numobs:)
    """
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
    """
    get constraint matrix for tile features
    assumes weak coupling. only 2 observations are constrained for each tile

    :param numfeat: number of features
    :param numobs: number of observations
    :return: constraint matrix (number of features, :param numobs:)
        each row is a constraint of form [index(A), lower bound A, upper bound A, index(B), lower bound B, upper bound B]
    """
    P = np.zeros([numfeat, 6])
    for feat in range(numfeat):
        for i in range(2):
            obs = random.choice(range(numobs))
            a = 2 * random.random() - 1
            if random.random() < 0.5:
                b = a + 0.4
            else:
                b = a - 0.4
            upper = max(a,b)
            lower = min(a,b)
            P[feat][i*3] = obs
            P[feat][i*3 + 1] = lower
            P[feat][i*3 + 2] = upper

    return P
        
def getP_rbf(numobs, grain = 3):
    """
    get matrix of equidistant means for radial basis functions

    :param numobs: number of observations
    :param grain: how many basis functions to generate for every dimension
    :return: matrix (number of features, :param numobs:)
    """
    elements = []
    for i in range(grain):
        elements.append(-1 + i* (2.0 / (grain - 1)))

    product = itertools.product(elements, repeat = numobs)
    P = list(product)
    return np.array(P)

def initialize_feature_parameters(num_features = 0, num_observations = 0, env = None, feature_type = "linear", sigma = 1, random = False):
    """
    construct feature parameters

    :param num_features: number of features, except for polynomials where it is the degree of the polynomial
    :param num_observations: number of observations
    :param env: gym learning environment
    :param feature_type: string specifying the kind of features to calculate - options are:
        "linear" - use observations as features
        "fourier" - fourier basis functions
        "rbf" - radial basis functions
        "polynomial" - polynomial basis functions
        "2dtiles" - (overlapping) tiles constrained in 2 dimensions
    :param sigma: standard deviation for rbf features
    :param random: boolean that determines if rbfs should be random or equidistant
    :return: feature parameters [(N,M), float, (N) or (M) or (0)] or None
        where N = :param num_features:
        and M = :param num_observations:
    """
    if feature_type == "linear":
        return
    elif feature_type == "polynomial":
        return [getP_polynomial(num_observations, num_features), 0, []]
    elif feature_type == "2dtiles":
        return [getP_2dtiles(num_features, num_observations), 0 , []]
    elif feature_type == "rbf":
        if not random:
            return [getP_rbf(num_observations), 0, sigma * np.ones(num_observations)]
        else:
            return [getP(num_features, num_observations, env = env), 0, sigma * np.ones(num_observations)]
    else:
        phi = getphi(num_features, env = env)
        P = getP(num_features, num_observations, env = env)
        v = 1
        return [P, v, phi]

def gradient_update(stepsize = 0.1, value = [], feature_params = [], x = 0, y = 0):
    """
    semi-gradient update for a linear value function

    :param stepsize: how far to move in the direction of the new parameter estimate (float)
    :param value: vector of value function parameters
    :param feature_params: feature parameters [P, 0, sigma]
    :param x: point at which to evaluate the value function (float)
    :param y: value objective to move towards (float)
    :return: vector of updated value parameters
    """
    features = radial_basis_functions(o = [x], p = feature_params)
    y_est = np.dot(value, features)
    value += stepsize * features * (y - y_est)

    return value

def plot(value = [], feature_params = [], validation_set = []):
    """
    plot the true function (a sinus) and its estimate

    :param value: vector of value function parameters
    :param feature_params: feature parameters [P, 0, sigma]
    :param validation_set: vector of points at which to evaluate the value function
    """
    y_est = []
    y_true = []
    for x in validation_set:
        y_true.append(np.sin(x))
        y_est.append(np.dot(value, radial_basis_functions(o = [x], p = feature_params)))

    plt.scatter(validation_set, y_true)
    plt.scatter(validation_set, y_est)
    plt.show()
    

def test_features():
    """
    learn to approximate a sinus function using rbf features and semi-gradient updates
    periodically plot the progress
    """
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
