import gym
import numpy
import random
import math

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

env = gym.make('Pendulum-v0')
pol = numpy.array([-1.70626311,  1.05841087,  1.06380155,  0.84091964,  0.82317395, 1.36914097, -0.13917266,  0.48150349, -0.30676511,  0.08903508, 0.05333251,  0.60618108])
totalr = 0
numfeat = 10
P = [[0.8271892692476548, 1.353786386812919, 0.6642739210288409], [0.26735626454778977, -1.247652519981915, -1.097728110377028], [-0.7878329915017503, 2.045007562286198, -1.2729253385127588], [-0.10630405283984809, -0.4902844285348495, -0.37888478211628435], [-1.8047990430079948, -0.049166395624201324, -1.447967462692155], [0.6965052119745367, 0.7767806101841989, 0.7002174711689194], [-1.080878855091235, 0.15471839918166952, 1.5108331779037636], [-0.17526413543551314, -2.0217598808803734, 0.8028825356595723], [-1.3928214119606517, -1.2183980388844198, 0.24273895450287258], [0.4381413033986252, 0.29218138102843594, 0.2628677031125169]]
v = 1
phi = [-2.4637767471536653, 0.31603378982007024, 2.1926405286563027, -2.813011639896043, 2.2505700662377137, 2.3460695921208288, -3.093408707337747, 2.1093803590050726, -2.4012236052580223, 2.907768673872578]



for _ in range(10):
    obs = env.reset()
    env.render()
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
        obs, r, done, info = env.step([action])
        env.render()
        reward += r
    print(reward)
    totalr += reward
env.close()   
print(totalr / 10.0)

