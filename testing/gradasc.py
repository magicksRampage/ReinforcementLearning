import gym
import math
import random
import numpy as np




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
    return np.array(y)

def derivative(act, feat, pol, mu):
    deriv = []
    sigma = pol[numfeat+1]
    for i in range(numfeat):
        d = feat[i] * (act - mu) / sigma
        deriv.append(d)
    deriv.append((act - mu) / sigma)
    deriv.append((1+(act - mu)*(act-mu) / sigma) / (2*sigma))
    return np.array(deriv)

def linearvalue(features, params, t, T):
    return np.dot(features, params) * (Tmax-t) / Tmax
    

env = gym.make('Pendulum-v0')
policy = []
trajectories = [] #t, s, a, r, dlp
batchsize = 100
numfeat = 10
numobs = 3
P = getP(numfeat,numobs)
v = 10
phi = getphi(numfeat)
omega = []
delta = 0.05
discount = 0.99
gaelambda = 0.90
Tmax = 200.
lrate = 0.05

for i in range(numfeat+2):
    policy.append(random.gauss(0,1))

policy = np.array(policy)
print(policy)
for i in range(numfeat):
    omega.append(random.gauss(0,1))
    #omega.append(0.)
omega = np.array(omega)

for gen in range(500):
    totalr = 0
    iterations = 100
    trajectories = []
   
    for i in range(iterations):
        traj = []
        obs = env.reset()
        done = False
        action = 0
        reward = 0
        t = 0
        while not done:
            feat = fourier(obs, P, v, phi)
            dot = np.dot(feat, policy[:numfeat])
            mu = dot + policy[numfeat]
            action = random.gauss(mu, policy[numfeat+1])
            a = 0
            if action > 1:
                a = 1
            #elif action < -1:
            #    a = -1
            else:
                a = 0
            obs, r, done, info = env.step([action])
            if i == 10:
               env.render()
            traj.append([t, feat, action, r, derivative(action, feat, policy, mu)])
            reward += r
            t += 1
        totalr += reward
        trajectories.append(traj)
        fishermat = np.zeros([numfeat+2, numfeat+2])
        gavg = 0
        fisheravg = np.zeros([numfeat+2, numfeat+2])
    
    newOmega = omega
    print(totalr / iterations)
    for traj in trajectories:
        g = 0
        fishermat = np.zeros([numfeat+2, numfeat+2])
        rev = range(len(traj))
        rev.reverse()
        for i in rev:
            Gt = traj[i][3]
            for j in range(i+1, len(traj)):
                Gt += math.pow(discount, j-i) * traj[j][3]
            traj[i].append(Gt)

        for i in range(len(traj)):
            sample = traj[i]
            tempdiff = 0
            value = linearvalue(sample[1], omega, sample[0], len(traj))
            if i == len(traj)-1:
                tempdiff = sample[3]
            else:
                samplep1 = traj[i+1]
                valuep1 = linearvalue(samplep1[1], omega, samplep1[0], len(traj))
                tempdiff = sample[3] + discount * valuep1 - value
            #print([lrate, sample[5], value, sample[1]])
            newOmega += lrate * (sample[5] - value) * sample[1] * (Tmax - sample[0]) / Tmax
            fishermat += np.transpose(np.mat(sample[4])) * np.mat(sample[4])
            traj[i].append(tempdiff)

        for i in range(len(traj)-1):
            advest = 0
            for j in range(i, len(traj)-1):
                sample = traj[j]
                advest += math.pow(discount * gaelambda, j-i) * sample[-1]
            g += traj[i][4] * advest
        gavg += g / len(traj)
        fisheravg += fishermat / len(traj)
    gavg /= iterations
    fisheravg /= iterations
    finverse = np.linalg.inv(fisheravg)    
    
    update = math.sqrt(delta / (np.mat(gavg) * finverse * np.transpose(np.mat(gavg)))) * finverse * np.transpose(np.mat(gavg))
    policy += np.array(np.transpose(update))[0]
    print(['policy', policy])
    omega = newOmega
    print(['value',omega])  

