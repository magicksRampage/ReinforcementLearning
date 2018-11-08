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

env = gym.make('CartPole-v0')
policies = []
performance = []
batchsize = 100
numfeat = 10
numobs = 4
P = getP(numfeat,numobs)
v = 1
phi = getphi(numfeat)
print([P, v, phi])

for _ in range(batchsize):
    w = []
    for i in range(numfeat+2):
        w.append(random.random() * 20 - 10)
    policies.append(numpy.array(w))
    
for gen in range(500):
    performance = []
    for pol in policies:
        totalr = 0
        iterations = 100
        for i in range(iterations):
            obs = env.reset()
            done = False
            action = 0
            reward = 0
            while not done:
                feat = fourier(obs, P, v, phi)
                dot = numpy.dot(feat, pol[:numfeat])
                dot += pol[numfeat]
                action = random.gauss(dot, pol[numfeat+1])
                a = 0
                if action > 0:
                    a = 1
                else:
                    a = 0
                obs, r, done, info = env.step(a)
                reward += r
            totalr += reward
        performance.append(totalr / iterations)

    polCopy = copyList(policies)
    perfCopy = copyList(performance)
    parents = []
    for i in range(batchsize):
        parent = [policies[i], performance[i]]
        parents.append(parent)

    parents.sort(key=sortByPerformance)
    #performance.sort()
    print([gen, parents[batchsize-1]])
    first = parents[0]
    for r in parents:
        r[1] -= first[1]
        
    newPols = []
    for i in range(1, batchsize):
        parents[i][1] += parents[i-1][1]
                
   
    for r in parents:
        r[1] /= parents[batchsize-1][1]

    for _ in range(batchsize-10):
        parent1 = getCeil(random.random(), parents)
        parent2 = getCeil(random.random(), parents)
        cutoff = int(math.ceil(random.random() * (numfeat + 1)))
        
        child = numpy.concatenate([parent1[:cutoff], parent2[cutoff:]])
        mutate = int(math.floor(random.random() * (numfeat + 2)))
        if random.random() < 0.1:
            child[mutate] = random.random() * 20 - 10
        newPols.append(child)
    policies = copyList(newPols)
    best = parents[90:]
    for b in best:
        policies.append(b[0])

    performance = []
