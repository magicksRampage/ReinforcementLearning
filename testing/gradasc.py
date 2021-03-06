import gym
import quanser_robots
import math
import random
import numpy as np
import matplotlib.pyplot as plt



def getphi(I):
    phi = []
    for i in range(I):
        phi.append(0)#random.random() * 2 * math.pi - math.pi)
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
#    return np.array(o)
#def realfourier():
    y = []
    for i in range(numfeat - len(y)):
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
        d = feat[i] * (act - mu) / (sigma * sigma)
        deriv.append(d)
    deriv.append((act - mu) / (sigma * sigma))
    deriv.append((- sigma +(act - mu)*(act-mu) / sigma) / (2*sigma*sigma))
    return np.array(deriv)

def linearvalue(features, params, t, T):
    featpt = np.append(features, 1)
    return np.dot(featpt, params)
    

env = gym.make('CartpoleStabShort-v0')
#env = gym.make('Pendulum-v0')
policy = []
trajectories = [] #t, s, a, r, dlp
batchsize = 100
numfeat = 30
numobs = env.observation_space.shape[0]
randomfeatures = False
if randomfeatures:
    P = getP(numfeat,numobs)
else:
    P = []
    P.append([1,0,0,0,0])
    P.append([0,1,0,0,0])
    P.append([0,0,1,0,0])
    P.append([0,0,0,1,0])
    P.append([0,0,0,0,1])
    P.append([1,1,0,0,0])
    P.append([1,0,1,0,0])
    P.append([1,0,0,1,0])
    P.append([1,0,0,0,1])
    P.append([0,1,1,0,0])
    P.append([0,1,0,1,0])
    P.append([0,1,0,0,1])
    P.append([0,0,1,1,0])
    P.append([0,0,1,0,1])
    P.append([0,0,0,1,1])
    P.append([2,0,0,0,0])
    P.append([0,2,0,0,0])
    P.append([0,0,2,0,0])
    P.append([0,0,0,2,0])
    P.append([0,0,0,0,2])
    P.append([3,0,0,0,0])
    P.append([0,3,0,0,0])
    P.append([0,0,3,0,0])
    P.append([0,0,0,3,0])
    P.append([0,0,0,0,3])
    P.append([4,0,0,0,0])
    P.append([0,4,0,0,0])
    P.append([0,0,4,0,0])
    P.append([0,0,0,4,0])
    P.append([0,0,0,0,4])
v = 0.2
phi = getphi(numfeat)
omega = []
delta = 0.0000001
discount = 0.99
gaelambda = 0.9
Tmax = 10000.
lrate = 0.05

for i in range(numfeat+2):
    policy.append(random.gauss(0,1))

policy = np.array(policy)
#policy[-1] = np.abs(1)
#policy[0] = -1
#policy[1] = 1000
#policy[2] = 0
#policy[3] = -1
#policy[4] = -1
#policy[5] = 0
print(policy)
for i in range(numfeat+1):
    omega.append(random.gauss(0,1))
    #omega.append(0.)
omega = np.array(omega)
#omega[2] = -10
print(omega)
avgRewards = []
render = False

for gen in range(5000):
    totalr = 0
    iterations = 0
    trajectories = []
    lrate = 0.1 /(gen+1)
    totalSamples = 0
    #for i in range(iterations):
    while totalSamples < 1000:
        traj = []
        obs = env.reset()
        done = False
        action = 0
        reward = 0
        t = 0
        if iterations == 0:
            render = True
        else:
            render = False
        while not done:
            feat = fourier(obs, P, v, phi)
            dot = np.dot(feat, policy[:numfeat])
            mu = dot + policy[numfeat]
            action = random.gauss(mu, policy[numfeat+1])
            #action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
            #action = obs[1] * 3000
            a = 0
            if action > 1:
                a = 1
            #elif action < -1:
            #    a = -1
            else:
                a = 0
            newobs, r, done, info = env.step(np.array([action]))
            #if render:
            #    env.render()
            traj.append([t, obs, action, r, newobs, derivative(action, feat, policy, mu)])
            totalSamples += 1
            reward += r
            t += 1
            obs = newobs
        totalr += reward
        trajectories.append(traj)
        iterations += 1
    fishermat = np.zeros([numfeat+2, numfeat+2])
    gavg = 0
    fisheravg = np.zeros([numfeat+2, numfeat+2])
    
    newOmega = omega.copy()
    print(['Generation',gen])
    print(['Avg Reward', totalr / iterations])
    avgRewards.append(totalr/iterations)
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    y0 = []
    y1 = []
    y2 = []
    y3 = []
    t_values = []
    t_features = []
    sample_no = 0
    if avgRewards[-1] > 500:
        render = True
    else:
        render = False
    for ti in range(len(trajectories)):
        traj = trajectories[ti]
        g = np.zeros([1, numfeat+2])[0]
        fishermat = np.zeros([numfeat+2, numfeat+2])
        rev = list(range(len(traj)))
        rev.reverse()
        for i in rev:
            Gt = traj[i][3]
            for j in range(i+1, len(traj)):
                Gt += math.pow(discount, j-i) * traj[j][3]
            traj[i].append(Gt)
        #print(traj)
        for i in range(len(traj)):
            sample = traj[i]
            sample_no += 1
            tempdiff = 0
            feat = fourier(sample[1], P, v, phi)
            value = linearvalue(feat, omega, sample[0], len(traj))
            value_new = linearvalue(feat, newOmega, sample[0], len(traj))
            #if np.abs(value) > 10000:
            #    print(str(value) + "," + str(feat) + "," + str(omega))
            x0.append(sample[1][0])
            x1.append(sample[1][1])
            x2.append(sample[1][2])
            #x3.append(sample[1][3])
            #x4.append(sample[1][4])
            t_values.append(sample[-1])
            t_features.append(feat)
            y0.append(value)
            y1.append(sample[2])
            y2.append(value_new)
            y3.append(sample[5][1])
            valuep1 = linearvalue(fourier(sample[4], P, v, phi), omega, sample[0], len(traj))
            valuep1_new = linearvalue(fourier(sample[4], P, v, phi), newOmega, sample[0], len(traj))
            tempdiff = sample[3] + discount * valuep1 - value
            tempdiff_new = sample[3] + discount * valuep1_new - value_new
            #tempdiff = sample[3] + discount * samplep1[-1] - sample[-1]
            #print([lrate, sample[5], value, sample[1]])
            dOmega =  (sample[-1] - value_new) * np.append(feat, 1)
            stepOmega = 50 / (500 + sample_no)# / np.dot(dOmega,dOmega)
            newOmega += stepOmega * dOmega
            fishermat += np.transpose(np.mat(sample[5])) * np.mat(sample[5])
            traj[i].append(tempdiff)
        if True:
            for i in range(len(traj)):
                advest = 0
                for j in range(i, len(traj)):
                    sample = traj[j]
                    advest += math.pow(discount * gaelambda, j-i) * sample[-1]
                g += traj[i][5] * advest
            gavg += g / len(traj)
            fisheravg += fishermat / len(traj)
    gavg /= iterations #/ 2
    fisheravg /= iterations #/ 2
    for line in fisheravg:
        for i in range(len(line)):
            if np.abs(line[i]) < 0.2:
                line[i] = line[i]

    #print(gavg)
    #print(fisheravg)
    #finverse = np.linalg.inv(fisheravg)
    #print(np.linalg.norm(fisheravg) * np.linalg.norm(finverse))
    #print(np.allclose(np.dot(fisheravg,finverse), np.eye(fisheravg.shape[0])))
    update = np.linalg.lstsq(fisheravg,gavg)
    update = update[0]
    #update = gavg
    #print(np.allclose(np.dot(fisheravg,update),gavg))
    stepsize = math.sqrt(delta / np.dot(update, gavg)) 
    #print(stepsize)
    #print(np.linalg.norm(update))
    policy += stepsize * update
    policy[-1] = np.abs(policy[-1])
    #print(update)
    print(['policy', policy])
    #omega = np.linalg.lstsq(t_features, t_values)[0]
    omega = newOmega.copy()
    #print(np.allclose(np.dot(t_features, omega), t_values))
    if np.mod(gen,100) == 0:
        #plt.scatter(x1, np.abs(np.array(y0) - t_values))
        plt.scatter(x1, t_values)
        #plt.semilogy(range(sample_no), np.abs(np.array(y2) - t_values))
        #plt.scatter(np.arange(len(omega)), omega)
        #plt.scatter(np.arange(len(omega)), newOmega)
        plt.scatter(x1, y0)
        plt.show()   
        plt.scatter(x1, y1)
        plt.show()
        #plt.scatter(x1, y3)
        #plt.show()
    print(['value',omega])  

plt.scatter(range(len(avgRewards)), avgRewards)
plt.show()
