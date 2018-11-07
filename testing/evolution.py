import gym
import math
import random
import numpy

env = gym.make('MountainCarContinuous-v0')
policies = []

for _ in range(100):
    w = []
    for i in range(3):
        w.append(random.random() * 20 - 10)
    policies.append(numpy.array(w))

performance = []
for pol in policies:
    totalr = 0
    iterations = 10
    for i in range(iterations):
        obs = env.reset()
        done = False
        action = 0
        reward = 0
        while not done:
            obs = numpy.concatenate([obs, numpy.array([1])])
            dot = numpy.dot(obs, pol)
            if dot > 0:
                action = 1
            else:
                action = 0
            obs, r, done, info = env.step([dot])
            reward += r
        totalr += reward
    performance.append(totalr / iterations)


for i in range(100):
    #if performance[i] > 0.0:
    print(policies[i])
    print(performance[i])
    
   
