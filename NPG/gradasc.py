import gym
import quanser_robots
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from quanser_robots import GentlyTerminating

import features


#global variables
delta = 0.05
discount = 0.9
gaelambda = 0.95


class RLstats:
    sines = None
    cosines = None
    avg_rewards = None
    sigmas = None
    est_values = None
    true_values = None
    


def getphi(I):
    phi = []
    for i in range(I):
        phi.append(random.random() * 2 * math.pi - math.pi)
    return phi


def getP(I,J):
    P = []
    limits = [0.407, 1, 1, 10, 10]
    for i in range(I):
        Pi = []
        for j in range(J):
            #Pi.append(random.gauss(0,1))
            Pi.append(2* limits[j] * random.random() - limits[j])
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


def get_features(observations, parameters, feature_type = "linear"):
    if feature_type == "linear":
        return observations
    elif feature_type == "fourier":
        return features.fourier(observations, parameters[0], parameters[1], parameters[2])
    elif feature_type == "rbf":
        return features.radial_basis_functions(observations, parameters)
    elif feature_type == "polynomial":
        return features.polynomial(observations, parameters)
    else:
        print("error: couldn't recognize feature type {}".format(feature_type))
        return observations

def derivative(act, feat, pol, mu):
    deriv = []
    sigma = pol[-1]
    for i in range(len(feat)):
        d = feat[i] * (act - mu) / (sigma * sigma)
        deriv.append(d)
    deriv.append((act - mu) / (sigma * sigma))
    deriv.append((- sigma +(act - mu)*(act-mu) / sigma) / (2*sigma*sigma))
    return np.array(deriv)

def linearvalue(features, params, t, T):
    featpt = np.append(features, 1)
    return np.dot(featpt, params)

def initialize_feature_parameters(num_features = 0, num_observations = 0):
    phi = getphi(num_features)
    P = getP(num_features, num_observations)
    v = 0.4
    
    return [P, v, phi]

def initialize_value_function(num_features = 0):
    omega = []
    for i in range(num_features+1):
        omega.append(random.gauss(0,1))
    omega = np.array(omega)
    return omega

def initialize_policy(num_features = 0):
    policy = []
    for i in range(num_features+2):
        policy.append(random.gauss(0,1))

    policy[-1] = np.abs(1)
    return policy

def generate_trajectories(min_iterations = 10, min_samples = 10000, env = None, policy = None, feature_params = None, feature_type = "linear", render_first = False):
    if env == None:
        print("error: please provide an environment")
        return []
    if feature_params == None and feature_type != "linear":
        print("error: please provide parameters for features")
        return []
    if policy == None:
        print("error: please provide a policy")
        return []

    numfeat = len(policy) -2
    totalr = 0
    miniterations = min_iterations
    trajectories = []
    totalSamples = 0
    while totalSamples < min_samples or len(trajectories) < miniterations:
        traj = []
        obs = env.reset()
        done = False
        action = 0
        reward = 0
        t = 0
        if render_first and iterations == 0:
            render = True
        else:
            render = False
        while not done:
            feat = get_features(obs, feature_params, feature_type)
            dot = np.dot(feat, policy[:numfeat])
            mu = dot + policy[numfeat]
            action = random.gauss(mu, policy[numfeat+1])

            newobs, r, done, info = env.step(np.array([action]))
            if render:
                env.render()
            traj.append([t, obs, action, r, newobs, derivative(action, feat, policy, mu)])
            totalSamples += 1
            reward += r
            t += 1
            obs = newobs
        totalr += reward
        trajectories.append(traj)

    return [trajectories, totalr]

def update_value_and_policy(trajectories = None, policy = None, value = None, feature_params = None, feature_type = "linear", lrate = 1):
    stats = RLstats()
    stats.sines = []
    stats.cosines = []
    stats.est_values = []
    stats.true_values = []
    stats.actions_taken = []

    numfeat = len(value) - 1
    iterations = len(trajectories)

    fishermat = np.zeros([numfeat+2, numfeat+2])
    gavg = 0
    fisheravg = np.zeros([numfeat+2, numfeat+2])
    featuremat = np.zeros([numfeat+1, numfeat+1])
    sumxv = 0
    omega = value
    newOmega = omega.copy()

    sample_no = 0
    for ti in range(len(trajectories)):
        traj = trajectories[ti]
        g = np.zeros([1, numfeat+2])[0]
        fishermat = np.zeros([numfeat+2, numfeat+2])
        totalrewards = list(range(len(traj)))
        rev = range(len(traj)-2,-1,-1)
        totalrewards[-1] = traj[-1][3]
        for i in rev:
            Gt = traj[i][3]
            Gt += discount * totalrewards[i+1]
            totalrewards[i] = Gt
        for i in range(len(traj)):
            sample = traj[i]
            sample_no += 1
            tempdiff = 0
            feat = get_features(sample[1], feature_params, feature_type)
            value = linearvalue(feat, omega, sample[0], len(traj))
            value_new = linearvalue(feat, newOmega, sample[0], len(traj))

            stats.sines.append(sample[1][1])
            stats.cosines.append(sample[1][2])
            stats.true_values.append(totalrewards[i])
            stats.est_values.append(value)
            stats.actions_taken.append(sample[2])

            featp1 = get_features(sample[4], feature_params, feature_type)
            valuep1 = linearvalue(featp1, omega, sample[0], len(traj))
            valuep1_new = linearvalue(featp1, newOmega, sample[0], len(traj))
            tempdiff = sample[3] + discount * valuep1 - value
            tempdiff_new = sample[3] + discount * valuep1_new - value_new
            mcdiff = totalrewards[i] - value_new
            feat = np.append(feat, 1)
            featp1 = np.append(featp1, 1)

            #Least  squares Monte Carlo
            #featuremat += np.transpose(np.mat(feat)) * np.mat(feat)
            #sumxv += np.array(feat) * totalrewards[i]

            #Least squares Temporal Difference
            featuremat += np.transpose(np.mat(feat)) * np.mat(feat - discount * featp1)
            sumxv += np.array(feat) * sample[3]

            fishermat += np.transpose(np.mat(sample[5])) * np.mat(sample[5])
            traj[i].append(tempdiff)

        # generalized advantage estimation
        advantages = list(range(len(traj)))
        advantages[-1] = traj[-1][-1]
        for i in rev:
            sample = traj[i]
            advest = sample[-1]
            advest += discount * gaelambda * advantages[i+1]
            advantages[i] = advest
            g += traj[i][5] * advest
        gavg += g / len(traj)
        fisheravg += fishermat / len(traj)
    gavg /= iterations #/ 2
    fisheravg /= iterations #/ 2
    newOmega = np.linalg.lstsq(featuremat,sumxv)[0] 
    update = np.linalg.lstsq(fisheravg,gavg)
    update = update[0]
    stepsize = math.sqrt(delta / np.dot(update, gavg)) 
    policy += stepsize * update
    policy = list(policy)
    print(['policy', policy])
    omega = lrate * newOmega + (1-lrate) * omega

    return [policy, omega, stats]

def main():
    env = GentlyTerminating(gym.make('CartpoleStabRR-v0'))
    policy = []
    trajectories = [] #t, s, a, r, dlp
    numobs = env.observation_space.shape[0]
    numfeat = numobs
    maxReward = 9999

    feature_params = initialize_feature_parameters(num_features = numfeat, num_observations = numobs)
    policy = initialize_policy(num_features = numfeat)
    omega = initialize_value_function(num_features = numfeat)

    print(policy)
    print(omega)
    avgRewards = []
    sigmas = np.array([])
    render = True

    for gen in range(5000):
        [trajectories, totalr] = generate_trajectories(env = env, min_iterations = 1, min_samples = 5000, policy = policy, feature_params = feature_params, feature_type = "linear")
        iterations = len(trajectories)

        print(['Generation',gen])
        print(['Avg Reward', totalr / iterations])
        avgRewards.append(totalr/iterations)
        sigmas = np.append(sigmas, policy[-1])

        lrate = 1.0 / (gen + 1)

        if avgRewards[-1] < maxReward:
            [policy, omega, stats] = update_value_and_policy(lrate = lrate, trajectories = trajectories, policy = policy, value = omega, feature_params = feature_params, feature_type = "linear")
        if np.mod(gen,50) == 0:
            plt.scatter(stats.sines, stats.true_values)
            plt.scatter(stats.sines, stats.est_values)
            plt.show()   
            plt.scatter(stats.sines, stats.actions_taken)
            plt.show()
            plt.scatter(range(len(avgRewards)), avgRewards)
            plt.scatter(range(len(sigmas)), 1000*sigmas)
            plt.show()
        print(['value',omega])  

main()
