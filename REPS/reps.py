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


def generate_episode(env_name, policy_name):
    """

    :param env_name:     designation of the environment the agent will act in and thus from which he gets his samples
    :param policy_name:  designation of the policy pursued while generating the episode
    :return: an array containing the sequence of samples generated in this episode
    """
    env = gym.make(env_name)
    prev_obs = env.reset()
    samples = ()
    action = 0
    lowAction = env.action_space.low[0]
    highAction = env.action_space.high[0]
    for i in range(0, 100):

        if policy_name == POLICY_RANDOM:
            action = random_policy(lowAction, highAction)

        elif policy_name == POLICY_GAUSSIAN:
            action = gaussian_policy(-1, lowAction, highAction, 0, 1)

        else:
            raise err.InvalidPolicyNameError

        obs, reward, done, info = env.step((action,))
        samples += ((prev_obs, action, obs, reward),)
        prev_obs = obs
        env.render()

    env.close()
    return samples


def evaluate_state_action_kernel(samples):
    """

    :param samples: set of state-action pairs available to calculate the state-action kernel
    :return: Matrix (len(samples))X(len(samples)) defining the state-action kernel
    """
    number_of_samples = np.shape(samples)[0]
    state_action_kernel = np.zeros((number_of_samples, number_of_samples))
    sample_i = None
    sample_j = None
    state_kval= 0.0
    action_kval = 0.0
    for i in range(0, number_of_samples):
        sample_i = samples[i]
        for j in range(0, number_of_samples):
            sample_j = samples[j]
            state_kval = gaussian_kernel(sample_i[0], sample_j[0])
            action_kval = gaussian_kernel(np.array([sample_i[1]]), np.array([sample_j[1]]))
            state_action_kernel[i][j] = state_kval * action_kval
    return state_action_kernel


def gaussian_kernel(vec1, vec2, bandwidth_matrix=None):
    """

    :param vec1: first argument in vector form (scalars might cause problems due to np.transpose())
    :param vec2: second argument in vector form (scalars might cause problems due to np.transpose())
    :param bandwidth_matrix: Matrix defining the free bandwith parameters of a gaussian kernel (must be len(vec1) == len(vec2)
    :return: scalar result of the kernel evaluation
    """
    dif_vec = vec1-vec2
    if bandwidth_matrix is None:
        bandwidth_matrix = np.identity(np.shape(dif_vec)[0])
    return np.exp(np.matmul(np.matmul(-np.transpose(dif_vec), bandwidth_matrix), dif_vec))


def calculate_beta(state, action, samples, K_sa, l_c=1.0):
    """

    :param state: state-argument put into beta(s,a)
    :param action: action-argument put into beta(s,a)
    :param samples: set of state-action pairs available to calculate beta(s,a)
    :param K_sa: the pre-calculated state-action kernel matrix
    :param l_c: regularization coefficient
    :return:
    """
    number_of_samples = np.shape(samples)[0]
    k_sa = np.zeros((number_of_samples, 1))
    sample = None
    for i in range(0,number_of_samples):
        sample = samples[i]
        state_kval = gaussian_kernel(sample[0], state)
        action_kval = gaussian_kernel(np.array([sample[1]]), np.array([action]))
        k_sa[i] = state_kval * action_kval
    reg_mat = np.multiply(l_c, np.identity(number_of_samples))
    beta = np.matmul(np.transpose(np.add(K_sa, reg_mat)),k_sa)
    return beta


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
    """ Which means prepare the Kernelmatrix so you can evaluate beta(s,a) """
    K_sa = evaluate_state_action_kernel(samples)
    test = calculate_beta(samples[0][0], samples[0][1], samples, K_sa)

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

    #K_s = evaluate_state_action_kernel()
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
