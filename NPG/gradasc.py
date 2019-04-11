import gym
import quanser_robots
import math
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt

import features

# global hyperparameters
# limits the stepsize of the policy update
delta = 0.05
# discount factor for value function
discount = 0.99
# lambda for generalized advantage estimation
gaelambda = 0.98


class RLstats:    
    """
    Contains statistics relating to the performance of a Reinforcement Learning Agent in list form
    """


    """
    stats for each step in a batch of episodes
    """
    # sine of the angle to be optimized
    sines = None
    
    # cosine of the angle to be optimized
    cosines = None

    # angle to be optimized
    thetas = None

    # value function
    est_values = None

    # discounted cumulative rewards
    true_values = None

    # action taken during each step
    actions_taken = None


    """
    stats for each batch of episodes
    """
    # difference between value function and discounted cumulative rewards averaged over a batch of episodes
    avg_value_loss = None

    # average reward of a batch of episodes
    avg_rewards = None

    # standard deviation of the policy
    sigmas = None

    # mean of the absolute values of the vanilla gradient
    vanilla_mean = None

    # variance of the absolute values of the vanilla gradient
    vanilla_variance = None

    # max of the absolute values of the vanilla gradient
    vanilla_max = None

    # min of the absolute values of the vanilla gradient
    vanilla_min = None

    # mean of the absolute values of the natural gradient
    update_mean = None

    # variance of the absolute values of the natural gradient
    update_variance = None

    # max of the absolute values of the natural gradient
    update_max = None

    # min of the absolute values of the natural gradient
    update_min = None

    # error in calculation of the natural gradient: fisher matrix * natural gradient - vanilla gradient
    update_error = None

    # rank of the fisher information matrix
    fisher_rank = None

    def __init__(self):
        """
        initializes each stat as an empty list
        """
        self.sines = []
        self.cosines = []
        self.thetas = []
        self.avg_rewards = []
        self.sigmas = []
        self.est_values = []
        self.true_values = []
        self.actions_taken = []
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
    """
    calculates the feature vector for a given state

    :param observations: vector of observations as supplied by the environment
    :param parameters: vector of parameters as supplied by features.initialize_feature_parameters
    :param feature_type: string specifying the kind of features to calculate - options are:
        "linear" - use observations as features
        "fourier" - fourier basis functions
        "rbf" - radial basis functions
        "polynomial" - polynomial basis functions
        "2dtiles" - (overlapping) tiles constrained in 2 dimensions
    :return: vector of features, length depends on :param parameters:
    """
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
    """
    calculates the derivative of the log of the policy with regard to the policy parameters for a given action

    :param act: action that was taken (float)
    :param feat: feature vector for the state the action was taken in
    :param pol: vector of policy parameters
    :param mu: mean policy for the state (float)
    :return: vector containing the derivative
    """
    deriv = []
    sigma = pol[-1]

    # derivation with respect to the feature parameters
    for i in range(len(feat)):
        d = feat[i] * (act - mu) / (sigma ** 2)
        deriv.append(d)

    # derivation with respect to the constant feature
    deriv.append((act - mu) / (sigma ** 2))

    # derivation with respect to sigma
    deriv.append((- sigma +(act - mu)**2 / sigma) / (2*sigma**2))
    return np.array(deriv)

def linearvalue(features, params, t, T):
    """
    calculates the value at a given state

    :param features: feature vector for the state
    :param params: vector of value function parameters
    :param t: timestep of the state - deprecated
    :param T: maximum episode length - deprecated
    :return: value for the given state (float)
    """
    featpt = np.append(features, 1)
    return np.dot(featpt, params)

def initialize_value_function(num_features = 0):
    """
    initializes value function parameters randomly

    :param num_features: number of features
    :return: vector of value function parameters (len = :param num_features: + 1)
    """
    omega = []

    # draw parameters from N(0,1)
    for i in range(num_features+1):
        omega.append(random.gauss(0,1))
    omega = np.array(omega)
    return omega

def initialize_policy(num_features = 0):
    """
    initializes policy parameters randomly

    :param num_features: number of features
    :return: vector of policy parameters (len = :param num_features: + 3)
    """
    policy = []

    # draw parameters from N(0,0.1)
    for i in range(num_features+2):
        policy.append(random.gauss(0,0.1))

    # set standard deviation
    policy[-1] = np.abs(3)
    return policy

def lstsqu_constraint(x, a, b):
    """
    equality constraint for minimization (:param a: * :param x: = :param b:)

    :param x: vector (N)
    :param a: matrix (N, N)
    :param b: vector (N)
    :return: difference between :param a: * :param x: and :param b:
    """
    return np.linalg.norm(b - np.dot(a,x))

def gradient_constraint(x, fisher):
    """
    inequality constraint for minimization (delta > :param x: * :param fisher: * :param x:)

    :param x: vector (N)
    :param fisher: matrix (N, N)
    :return: difference between delta and :param x: * :param fisher: * :param x:
    """
    return delta - np.dot(x, np.dot(fisher, x))

def variance_constraint(x, y):
    """
    inequality constraint for minimization (var(:param x:) > 1)

    :param x: vector (N)
    :param y: vector (N) - deprecated
    :return: difference between var(:param x:) and 1
    """
    return np.var(x) - 1

def max_constraint(x):
    """
    inequality constraint for minimization (max(abs(:param x:)) > 10)

    :param x: vector (N)
    :return: difference between max(abs((:param x:))) and 10
    """
    return max_objective(x) - 10

def max_objective(x):
    """
    objective function for minimization (max(abs(:param x:)))

    :param x: vector (N)
    :return: max(abs(:param x:))
    """
    return max(np.abs(x))

def generate_trajectories(min_iterations = 10, min_samples = 10000, env = None, policy = None, feature_params = None, feature_type = "linear", render_first = False):
    """
    generate trajectories by interacting with the environment

    :param min_iterations: minimum number of episodes to be generated (int)
    :param min_samples: minimum number of steps to take (int)
    :param env: gym learning environment
    :param policy: vector of policy parameters used to generate the trajectories
    :param feature_params: parameters to calculate the features ([P, v, phi])
    :param feature_type: string specifying the kind of features to calculate - options are:
        "linear" - use observations as features
        "fourier" - fourier basis functions
        "rbf" - radial basis functions
        "polynomial" - polynomial basis functions
        "2dtiles" - (overlapping) tiles constrained in 2 dimensions
    :param render_first: boolean that decides whether to render the first episode of the batch
    :return trajectories: list of trajectories each containing a list of samples of form ([timestep, observations, action, reward, next observations, derivative of log policy wrt policy :param policy:])
    :return totalr: list of undiscounted total rewards for each trajectory
    """

    # exception handling
    if env == None:
        print("error: please provide an environment")
        return []
    if feature_params == None and feature_type != "linear":
        print("error: please provide parameters for features")
        return []
    if policy == None:
        print("error: please provide a policy")
        return []

    # initialize variables
    numfeat = len(policy) -2
    totalr = 0
    miniterations = min_iterations
    trajectories = []
    totalSamples = 0
    limits = np.array(env.observation_space.high)

    # start new episode while there aren't enough samples or episodes yet
    while totalSamples < min_samples or len(trajectories) < miniterations:

        # initialize new episode
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

        # call env.step until the environment returns the done flag
        while not done:

            # normalize observations to [-1, 1]
            obs = np.array(obs) / limits

            # calculate action
            feat = get_features(obs, feature_params, feature_type)
            dot = np.dot(feat, policy[:numfeat])
            mu = dot + policy[numfeat]
            action = random.gauss(mu, policy[numfeat+1])

            # take step in the environment
            newobs, r, done, info = env.step(np.array([action]))
            if render:
                env.render()

            # save sample in trajectory
            traj.append([t, obs, action, r, newobs, derivative(action, feat, policy, mu)])
            totalSamples += 1
            reward += r
            t += 1
            obs = newobs

        # increase total reward by episode reward
        totalr += reward
        trajectories.append(traj)

    return [trajectories, totalr]

def update_value_and_policy(trajectories = None, policy = None, value = None, feature_params = None, feature_type = "linear", lrate = 1, stats = None):
    """
    evaluate a given number of trajectories
    update value function using Least-Squares Temporal Difference Learning
    update policy using Natural Policy Gradients

    :param trajectories: list of trajectories each containing a list of samples of form ([timestep, observations, action, reward, next observations, derivative of log policy wrt policy :param policy:])
    :param policy: vector of policy parameters to be updated
    :param policy: vector of value function parameters to be updated
    :param feature_params: parameters to calculate the features ([P, v, phi])
    :param feature_type: string specifying the kind of features to calculate - options are:
        "linear" - use observations as features
        "fourier" - fourier basis functions
        "rbf" - radial basis functions
        "polynomial" - polynomial basis functions
        "2dtiles" - (overlapping) tiles constrained in 2 dimensions
    :param lrate: learning rate for the value function update
    :param stats: RLstats object to record learning statistics
    :return policy: updated policy parameters
    :return omega: updated value function parameters
    :return stats: RLstats object containing learning statistics
    """

    # initialize stats
    if stats is None:
        stats = RLstats()

    stats.sines = []
    stats.cosines = []
    stats.thetas = []
    stats.est_values = []
    stats.true_values = []
    stats.actions_taken = []

    # initialize parameters
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
    # loop over all trajectories
    for ti in range(len(trajectories)):

        # initialize data structures for new trajectory
        traj = trajectories[ti]
        g = np.zeros([numfeat+2])
        fishermat = np.zeros([numfeat+2, numfeat+2])
        totalrewards = list(range(len(traj)))
        rev = range(len(traj)-2,-1,-1)
        totalrewards[-1] = traj[-1][3]

        # calculate discounted rewards for each timestep
        for i in rev:
            Gt = traj[i][3]
            Gt += discount * totalrewards[i+1]
            totalrewards[i] = Gt

        # loop over all samples in a trajectory
        for i in range(len(traj)):
            # initialize sample
            sample = traj[i]
            sample_no += 1
            tempdiff = 0
            feat = get_features(sample[1], feature_params, feature_type)
            value = linearvalue(feat, omega, sample[0], len(traj))
            value_new = linearvalue(feat, newOmega, sample[0], len(traj))

            # save stats
            stats.sines.append(sample[1][1])
            stats.cosines.append(sample[1][0])
            stats.thetas.append(np.arctan2(sample[1][1], sample[1][0]))
            stats.true_values.append(totalrewards[i])
            stats.est_values.append(value)
            stats.actions_taken.append(sample[2])

            # calculate temporal difference
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

            # calculate the fisher information matrix
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
            # estimate the gradient as derivative of log policy * advantage
            g += traj[i][5] * advest

        # average the gradient and fisher matrix over all samples
        gavg += g / len(traj)
        fisheravg += fishermat / len(traj)

    # average the gradient
    gavg /= iterations #/ 2
    fisheravg /= iterations #/ 2

    # least squares temporal difference for value update
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
    # solve system of linear equations for the natural gradient
    [update, residuals, rank, singular] = np.linalg.lstsq(fisheravg,gavg,rcond = 0.00002)
    #opt_result = scipy.optimize.minimize(fun = np.dot, args = -gavg, x0 = update, method = "SLSQP", constraints = gradient_constraints)
    #opt_result = scipy.optimize.minimize(fun = np.var, x0 = update, method = "SLSQP", constraints = var_constraints)
    #update = opt_result.x

    # normalized stepsize for policy update
    stepsize = math.sqrt(delta / np.abs(np.dot(update, gavg)))
    policy += stepsize * update
    policy = list(policy)

    # value update
    omega = lrate * newOmega + (1-lrate) * omega

    # save stats
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
    """
    main method

    sets up an environment and learning parameters and proceeds to learn a policy using Natural Policy Gradients
    plots stats related to the learning progress every 50 iterations
    """

    # initialize environment
    env = gym.make('Qube-v0')
    policy = []
    trajectories = [] #t, s, a, r, s', dln(p)
    numobs = env.observation_space.shape[0]

    # set hyperparameters
    numfeat = numobs
    feature_type = "linear"
    features_random = False
    sigma_rbf = 20
    render_first = False
    maxReward = 10 ** 10

    stats = RLstats()

    # initialize feature parameters, policy and value function
    feature_params = features.initialize_feature_parameters(num_features = numfeat, num_observations = numobs, env = env, feature_type = feature_type, sigma = sigma_rbf, random = features_random)
    policy = initialize_policy(num_features = numfeat)
    omega = initialize_value_function(num_features = numfeat)

    print(policy)
    print(omega)

    # repeat learning process arbitrarily often
    for gen in range(5000):

        # generate trajectories
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

        # update value function and policy
        if stats.avg_rewards[-1] < maxReward:
            [policy, omega, stats] = update_value_and_policy(lrate = lrate, trajectories = trajectories, policy = policy, value = omega, feature_params = feature_params, feature_type = feature_type, stats = stats)
        print(['policy', policy])

        # plot results every 50 iterations
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
    """
    compares learning performance of different hyperparameters
    plots value loss and reward progression over 20 iterations
    """
    stats_map = {}
    for lamb in [0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
        global gaelambda
        gaelambda = lamb
        env = gym.make('Qube-v0')
        policy = []
        trajectories = [] #t, s, a, r, s', dln(p)
        numobs = env.observation_space.shape[0]
        numfeat = numobs
        feature_type = "linear"
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

main()
#compare_features()
