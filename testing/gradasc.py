import gym
import math
import random
import numpy




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
    

def sortByPerformance(par):
    return par[1]

def getCeil(query,l):
    for i in range(len(l)):
        if query < l[i][1]:
            return l[i][0]
    
def copyList(l):
    copy = []
    for i in range(len(l)):
        copy.append(l[i])
    return copy

def fourier(o, P, v, phi):
    y = []
    for i in range(numfeat):
        arg = 0
        for j in range(len(o)):
            arg += P[i][j] * o[j]
        arg /= v
        arg += phi[i]
        y.append(math.sin(arg))
    return numpy.array(y)

def derivative(act, feat, pol, mu):
    deriv = []
    sigma = feat[numfeat+1]
    for i in range(numfeat):
        d = pol[i] * (act - mu) / sigma
        deriv.append(d)
    deriv.append((act - mu) / sigma)
    deriv.append((1+(act - mu)*(act-mu) / sigma) / (2*sigma))
    return deriv

env = gym.make('MountainCarContinuous-v0')
policy = []
trajectories = [] #t, s, a, r
batchsize = 100
numfeat = 10
numobs = 2
P = getP(numfeat,numobs)
v = 1
phi = getphi(numfeat)

for i in range(numfeat+2):
    policy.append(random.random() * 20 - 10)
    
for gen in range(500):
    totalr = 0
    iterations = 100
    for i in range(iterations):
        traj = []
        obs = env.reset()
        done = False
        action = 0
        reward = 0
        t = 0
        while not done:
            feat = fourier(obs, P, v, phi)
            dot = numpy.dot(feat, pol[:numfeat])
            mu = dot + pol[numfeat]
            action = random.gauss(mu, pol[numfeat+1])
            a = 0
            if action > 0:
                a = 1
            else:
                a = 0
            obs, r, done, info = env.step([action])
            traj.append([t, feat, action, r])
            reward += r
            t++
        totalr += reward
        trajectories.append(traj)

