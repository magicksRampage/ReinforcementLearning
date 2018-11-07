import gym
import numpy

env = gym.make('MountainCarContinuous-v0')
pol = numpy.arraypol = numpy.array([-0.707, 6.423])
totalr = 0

for _ in range(10):
    obs = env.reset()
    env.render()
    done = False
    action = 0
    reward = 0
    while not done:
        dot = numpy.dot(obs, pol)
        if dot > 0:
            action = 1
        else:
            action = 0
        obs, r, done, info = env.step([dot])
        env.render()
        reward += r
    print(reward)
    totalr += reward

#print(totalr / 1000.0)

