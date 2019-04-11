import gym
import quanser_robots
import math
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt

import features

#global variables
delta = 0.05
discount = 0.99
gaelambda = 0.97


class RLstats:
    sines = None
    cosines = None
    thetas = None
    avg_rewards = None
    sigmas = None

    est_values = None
    true_values = None
    avg_value_loss = None

    vanilla_mean = None
    vanilla_variance = None
    vanilla_max = None
    vanilla_min = None

    update_mean = None
    update_variance = None
    update_max = None
    update_min = None

    update_error = None
    fisher_rank = None
    def __init__(self):
        self.sines = []
        self.cosines = []
        self.thetas = []
        self.avg_rewards = []
        self.sigmas = []
        self.est_values = []
        self.true_values = []
        self.avg_value_loss = []
        self.vanilla_mean = []
        self.vanilla_variance = []
        self.vanilla_max = []
        self.vanilla_min = []
        self.update_mean = []
        self.update_variance = []
        self.update_max = []
        self.update_min = []
        self.update_error = []
        self.fisher_rank = []


def get_features(observations, parameters, feature_type = "linear"):
    if feature_type == "linear":
        return observations
    elif feature_type == "fourier":
        return features.fourier(observations, parameters[0], parameters[1], parameters[2])
    elif feature_type == "rbf":
        return features.radial_basis_functions(observations, parameters)
    elif feature_type == "polynomial":
        return features.polynomial(observations, parameters)
    elif feature_type == "2dtiles":
        return features.tiling(observations, parameters)
    else:
        print("error: couldn't recognize feature type {}".format(feature_type))
        return observations

def derivative(act, feat, pol, mu):
    deriv = []
    sigma = pol[-1]
    for i in range(len(feat)):
        d = feat[i] * (act - mu) / (sigma ** 2)
        deriv.append(d)
    deriv.append((act - mu) / (sigma ** 2))
    deriv.append((- sigma +(act - mu)**2 / sigma) / (2*sigma**2))
    return np.array(deriv)

def linearvalue(features, params, t, T):
    featpt = np.append(features, 1)
    return np.dot(featpt, params)

def initialize_value_function(num_features = 0):
    omega = []
    for i in range(num_features+1):
        omega.append(random.gauss(0,1))
    omega = np.array(omega)
    return omega

def initialize_policy(num_features = 0):
    policy = []
    for i in range(num_features+2):
        policy.append(random.gauss(0,0.1))
    policy[-1] = np.abs(3)
    return policy

def lstsqu_constraint(x, a, b):
    return np.linalg.norm(b - np.dot(a,x))

def gradient_constraint(x, fisher):
    return delta - np.dot(x, np.dot(fisher, x))

def variance_constraint(x, y):
    return np.var(x) - 1

def max_constraint(x):
    return max_objective(x) - 10

def max_objective(x):
    return max(np.abs(x))

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
    limits = np.array(env.observation_space.high)
    while totalSamples < min_samples or len(trajectories) < miniterations:
        traj = []
        obs = env.reset()
        done = False
        action = 0
        reward = 0
        t = 0
        if render_first and len(trajectories) == 0:
            render = True
        else:
            render = False
        while not done:
            obs = np.array(obs) / limits

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

def update_value_and_policy(trajectories = None, policy = None, value = None, feature_params = None, feature_type = "linear", lrate = 1, stats = None):
    if stats is None:
        stats = RLstats()

    stats.sines = []
    stats.cosines = []
    stats.thetas = []
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
        g = np.zeros([numfeat+2])
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
            stats.cosines.append(sample[1][0])
            stats.thetas.append(np.arctan2(sample[1][1], sample[1][0]))
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
    newOmega = np.linalg.lstsq(featuremat,sumxv, rcond = None)[0]
    var_constraints = [
        {
            "type":"eq",
            "fun":lstsqu_constraint,
            "args": [fisheravg, gavg]
        }
    ]
    gradient_constraints = [
        {
            "type":"ineq",
            "fun":gradient_constraint,
            "args": [fisheravg]
        }
    ]
    [update, residuals, rank, singular] = np.linalg.lstsq(fisheravg,gavg,rcond = 0.00002)
    #opt_result = scipy.optimize.minimize(fun = np.dot, args = -gavg, x0 = update, method = "SLSQP", constraints = gradient_constraints)
    #opt_result = scipy.optimize.minimize(fun = np.var, x0 = update, method = "SLSQP", constraints = var_constraints)
    #update = opt_result.x
    stepsize = math.sqrt(delta / np.abs(np.dot(update, gavg)))
    #update = gavg
    #stepsize = delta / max(update)
    policy += stepsize * update
    policy = list(policy)
    omega = lrate * newOmega + (1-lrate) * omega

    value_diff = np.array(stats.est_values) - np.array(stats.true_values)
    total_loss = np.linalg.norm(value_diff)
    relative_loss = total_loss / np.linalg.norm(stats.true_values)
    avg_loss = relative_loss / len(stats.true_values)
    stats.avg_value_loss.append(avg_loss)

    stats.vanilla_mean.append(sum(np.abs(gavg)) / len(gavg))
    stats.vanilla_variance.append(np.var(np.abs(gavg)))
    stats.vanilla_max.append(max(np.abs(gavg)))
    stats.vanilla_min.append(min(np.abs(gavg)))

    stats.update_mean.append(sum(np.abs(update)) / len(update))
    stats.update_variance.append(np.var(np.abs(update)))
    stats.update_max.append(max(np.abs(update)))
    stats.update_min.append(min(np.abs(update)))

    stats.update_error.append(np.linalg.norm(gavg - np.dot(fisheravg, update)) / len(gavg))
    stats.fisher_rank.append(rank)

    return [policy, omega, stats]

def main():
    env = gym.make('Qube-v0')
    policy = []
    trajectories = [] #t, s, a, r, s', dln(p)
    numobs = env.observation_space.shape[0]
    numfeat = 729
    feature_type = "rbf"
    features_random = False
    sigma_rbf = 20
    render_first = False
    maxReward = 10 ** 10

    stats = RLstats()

    feature_params = features.initialize_feature_parameters(num_features = numfeat, num_observations = numobs, env = env, feature_type = feature_type, sigma = sigma_rbf, random = features_random)
    policy = initialize_policy(num_features = numfeat)
    omega = initialize_value_function(num_features = numfeat)

    print(policy)
    print(omega)

    for gen in range(5000):
        [trajectories, totalr] = generate_trajectories(render_first = render_first, env = env, min_iterations = 5, min_samples = 1000, policy = policy, feature_params = feature_params, feature_type = feature_type)
        iterations = len(trajectories)

        print(['Generation',gen])
        print(['Avg Reward', totalr / iterations])
        stats.avg_rewards.append(totalr/iterations)
        stats.sigmas.append(policy[-1])

        global delta
        if gen == 0:
            delta = 0
        else:
            delta = 0.05

        lrate = 1.0

        if stats.avg_rewards[-1] < maxReward:
            [policy, omega, stats] = update_value_and_policy(lrate = lrate, trajectories = trajectories, policy = policy, value = omega, feature_params = feature_params, feature_type = feature_type, stats = stats)
        print(['policy', policy])

        if np.mod(gen+1,50) == 0:
            plt.scatter(stats.thetas, stats.true_values)
            plt.scatter(stats.thetas, stats.est_values)
            plt.legend(['true values', 'estimated values'])
            plt.show()   
            plt.semilogy(range(len(stats.avg_value_loss)), stats.avg_value_loss)
            plt.legend(['average value loss'])
            plt.show()
            plt.scatter(stats.thetas, stats.actions_taken)
            plt.legend(['actions taken'])
            plt.show()
            plt.plot(stats.avg_rewards)
            plt.plot(stats.sigmas)
            plt.legend(['average rewards', 'sigma (policy)'])
            plt.show()
            plt.semilogy(stats.update_variance)
            plt.semilogy(stats.vanilla_variance)
            plt.legend(['natural gradient variance', 'vanilla gradient variance'])
            plt.show()
            plt.semilogy(stats.update_error)
            plt.legend(['natural gradient error'])
            plt.show()
            plt.plot(stats.fisher_rank)
            plt.legend(['rank of fisher matrix'])
            plt.show()
        print(['value',omega])  

def compare_features():
    stats_map = {}
    for lamb in [0.97]:
        global gaelambda
        gaelambda = lamb
        env = gym.make('Qube-v0')
        policy = []
        trajectories = [] #t, s, a, r, s', dln(p)
        numobs = env.observation_space.shape[0]
        numfeat = 729
        feature_type = "rbf"
        sigma_rbf = 20
        render_first = False
        maxReward = 10 ** 10

        stats = RLstats()

        feature_params = features.initialize_feature_parameters(num_features = numfeat, num_observations = numobs, env = env, feature_type = feature_type, sigma = sigma_rbf)
        policy = initialize_policy(num_features = numfeat)
        omega = initialize_value_function(num_features = numfeat)


        for gen in range(20):
            [trajectories, totalr] = generate_trajectories(render_first = render_first, env = env, min_iterations = 1, min_samples = 10000, policy = policy, feature_params = feature_params, feature_type = feature_type)
            iterations = len(trajectories)

            stats.avg_rewards.append(totalr/iterations)
            stats.sigmas.append(policy[-1])

            lrate = 1.0

            if stats.avg_rewards[-1] < maxReward:
                [policy, omega, stats] = update_value_and_policy(lrate = lrate, trajectories = trajectories, policy = policy, value = omega, feature_params = feature_params, feature_type = feature_type, stats = stats)

                #print("len(true_values): {}, value_diff: {}, relative_diff: {}, relative_loss: {}, avg_loss: {}".format(len(stats.true_values), value_diff, relative_diff, relative_loss, avg_loss))
        stats_map[str(lamb)] = stats

    for key in stats_map:
        stats = stats_map[key]
        print(key)
        avg_last_10 = sum(stats.avg_value_loss[10:]) / 10
        print("average value loss: {}".format(avg_last_10))
        print("average reward: {}".format(stats.avg_rewards[-1]))
        plt.plot(stats.avg_rewards)
    plt.legend(list(stats_map.keys()))
    plt.show()

    for key in stats_map:
        stats = stats_map[key]
        plt.semilogy(stats.avg_value_loss)
    plt.legend(list(stats_map.keys()))
    plt.show()

#main()
compare_features()
