import gym
import quanser_robots
import numpy as np
import errors as err

POLICY_GAUSSIAN = 'gaussian'
POLICY_RANDOM = 'random'

ENV_PENDULUM = 'Pendulum-v0'


def random_policy(space_low, space_high):
    """

    :param space_low:  low end of the action space
    :param space_high:   high end of the action space
    :return: returns a sample from a equiprobable distribution over the action space
    """

    space_range = space_high - space_low
    sample = space_range * np.random.random() + space_low
    return sample


def gaussian_policy(state, space_low, space_high, loc, scale):
    """
    
    :param state:       state in which the agent has to act
    :param space_low:   low end of the action space
    :param space_high:  high end of the action space
    :param loc:         mean of the gaussian
    :param scale:       standard deviation of the gaussian
    :return: a sample from a gaussian distribution over the action space
    """

    space_range = space_high - space_low
    sample = np.random.normal(loc, scale)
    return np.clip(sample, space_low, space_high)


def generate_episode(envName, policyName):
    """

    :param envName:     designation of the environment the agent will act in and thus from which he gets his samples
    :param policyName:  designation of the policy pursued while generating the episode
    :return: an array containing the sequence of samples generated in this episode
    """
    env = gym.make(envName)
    env.reset()
    samples = ()
    action = 0
    lowAction = env.action_space.low[0]
    highAction = env.action_space.high[0]
    for i in range(1, 100):

        if policyName == POLICY_RANDOM:
            action = random_policy(lowAction, highAction)

        elif policyName == POLICY_GAUSSIAN:
            action = gaussian_policy(-1, lowAction, highAction, 0, 1)

        else:
            raise err.InvalidPolicyNameError

        obs, reward, done, info = env.step((action,))
        samples += ((obs, reward),)
        env.render()

    env.close()
    return samples


def evaluate_kernel(samples):
    # TODO: implement
    # TODO: diffentiate between K_sa and K_s
    K_sa_ij = k_s(s_i,s)*k_a(a_i,a)
    return np.zeros(0)


def fast_minimization():
    # TODO: implement
    return 0.0


def constrained_minimization():
    # TODO: implement
    return 0.0


def update_step(old_policy):

    """ 1. generate roll-outs according to pi_(i-1) """

    samples = generate_episode(ENV_PENDULUM, POLICY_GAUSSIAN)

    """ 2. calculate kernel embedding strengths """

    beta = np.zeros(0)
    K_sa = evaluate_kernel(samples)
    # TODO: Extract k_sa as column from K_sa
    k_sa = np.zeros(0)
    l_C = 0.0
    # TODO: Is I the identity matrix?
    I = np.zeros(0)
    #beta = np.matmul(np.linalg.inv(np.add(K_sa, l_C * I)), k_sa)

    """ 3. minimize kernel-based dual """

    # iterate coordinate-descent (till constraints are sufficiently fulfilled)
    alpha = np.inf
    eta = np.inf
    # convergence threshold
    epsilon = 0.1
    while (alpha > epsilon) | (epsilon > epsilon):

        # fast unconstrained convex optimization on alpha (fixed iterations)
        for iteration in range(0,10):
            alpha = fast_minimization()

        # constrained minimization on eta (fixed iterations)
        for iteration in range(0,10):
            eta = constrained_minimization()

    """ 4. calculate kernel-based Bellman errors """

    K_s = evaluate_kernel()
    # TODO: Extract k_s as a column from K_s
    k_s = np.zeros(0)
    # TODO: Where do we get R_as from?
    R_as = np.zeros(0)
    # TODO: Gather all the deltas
    #delta_j = np.add(R_as, np.matmul(np.transpose(alpha), np.subtract(np.matmul(K_s, beta), k_s)))

    """ 5. calculate the sample weights """

    # TODO: Gather all the weights
    #w_j = np.exp(delta_j / eta)

    """ 6. fit a generalizing non-parametric policy"""

    # hyper-parameter
    l = 1
    # TODO: diagonal matrix with D_ii = 1 / w_i
    D = np.zeros(0)
    # TODO: A = [a_1, ..., a_n]^T
    A = np.zeros(0)
    #mu = np.matmul(np.matmul(np.transpose(k_s), np.linalg.inv(np.add(K_s, l * D))), A)
    mu = 0
    sigma_sq = 0.0
    new_policy = (mu, sigma_sq)

    return new_policy


def main():

    converged = False
    #Parameters for a gaussian policy (mu, sigma_sq)
    policy = (0, 1)
    while not converged:
        policy = update_step(policy)
        # TODO: define conversion target
        converged = True


main()
