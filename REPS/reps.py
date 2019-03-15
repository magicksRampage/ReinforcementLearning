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


def evaluate_state_transition_kernel(samples):
    """

    :param samples: set of state-action pairs available to calculate the state-transition kernel
    :return: Matrix (len(samples))X(len(samples)) defining the state-transition kernel
    """
    number_of_samples = np.shape(samples)[0]
    state_transition_kernel = np.zeros((number_of_samples, number_of_samples))
    sample_i = None
    sample_j = None
    for i in range(0, number_of_samples):
        sample_i = samples[i]
        for j in range(0, number_of_samples):
            sample_j = samples[j]
            state_transition_kernel[i][j] = gaussian_kernel(sample_i[0], sample_j[2])
    return state_transition_kernel


def gaussian_kernel(vec1, vec2, bandwidth_matrix=None):
    """

    :param vec1: first argument in array form (scalars will cause problems due to np.transpose())
    :param vec2: second argument in array form (scalars will cause problems due to np.transpose())
    :param bandwidth_matrix: Matrix defining the free bandwidth parameters of a gaussian kernel (must be len(vec1) == len(vec2)
    :return: scalar result of the kernel evaluation
    """
    dif_vec = vec1-vec2
    if bandwidth_matrix is None:
        bandwidth_matrix = np.identity(np.shape(dif_vec)[0])
    return np.exp(np.matmul(np.matmul(-np.transpose(dif_vec), bandwidth_matrix), dif_vec))


def calculate_beta(state, action, samples, transition_kernel, l_c=1.0):
    """

    :param state: state-argument for which to evaluate beta(s,a)
    :param action: action-argument for which to evaluate beta(s,a)
    :param samples: set of state-action pairs available to calculate beta(s,a)
    :param transition_kernel: the pre-calculated state-action kernel matrix
    :param l_c: regularization coefficient
    :return: beta(s,a)
    """
    number_of_samples = np.shape(samples)[0]
    state_action_vec = np.zeros((number_of_samples, 1))
    sample = None
    for i in range(0,number_of_samples):
        sample = samples[i]
        state_kval = gaussian_kernel(sample[0], state)
        action_kval = gaussian_kernel(np.array([sample[1]]), np.array([action]))
        state_action_vec[i] = state_kval * action_kval
    reg_mat = np.multiply(l_c, np.identity(number_of_samples))
    beta = np.matmul(np.transpose(np.add(transition_kernel, reg_mat)), state_action_vec)
    return beta


def calculate_embedding_vector(state, action, samples, transition_kernel):
    """

    :param state: state-argument for which to evaluate the embedding vector
    :param action: action-argument for which to evaluate the embedding vector
    :param samples: set of state-action pairs available to calculate the embedding vector
    :param transition_kernel: the pre-calculated state-action kernel matrix
    :param l_c: regularization coefficient
    :return: the embedding vector (s,a) needed for the bellman-error
    """

    beta = calculate_beta(state, action, samples, transition_kernel)

    number_of_samples = np.shape(samples)[0]
    transition_vec = np.zeros((number_of_samples, 1))
    sample = None
    for i in range(0, number_of_samples):
        sample = samples[i]
        transition_vec[i] = gaussian_kernel(sample[0], state)

    embedding_vec = np.add(np.matmul(transition_kernel, beta), -transition_vec)
    return embedding_vec


def calculate_bellman_error(reward, alpha, embedding_vector):
    """

    :param reward:
    :param alpha:
    :param embedding_vector:
    :return:
    """
    # TODO: Comment
    error = reward + np.matmul(np.transpose(alpha), embedding_vector)
    return error


def minimize_dual_for_alpha(initial_alpha, eta, samples, number_of_iterations=10):
    """

    :param initial_alpha:
    :param eta:
    :param samples:
    :param number_of_iterations:
    :return:
    """
    # TODO: Comment
    number_of_samples = np.shape(samples)[0]
    transition_kernel = evaluate_state_transition_kernel(samples)
    alpha = initial_alpha

    embedding_vectors = np.zeros((number_of_samples, number_of_samples))
    for i in range(0, number_of_samples):
        sample = samples[i]
        embedding_vectors[:, i] = calculate_embedding_vector(sample[0], sample[1], samples, transition_kernel).reshape(number_of_samples,)

    # Initialize values but once
    bellman_errors = np.zeros((number_of_samples, 1))
    temp = np.zeros((number_of_samples, 1))
    weights = np.zeros((number_of_samples, 1))
    denominator = None
    partial = np.zeros((number_of_samples, 1))

    for descent in range(0, number_of_iterations):

        sample = None
        for i in range(0, number_of_samples):
            sample = samples[i]
            bellman_errors[i] = calculate_bellman_error(sample[3], alpha, embedding_vectors[:, i])

        for i in range(0, number_of_samples):
            denominator = 0.0
            for j in range(i, number_of_samples):
                denominator += np.exp(np.divide(bellman_errors[j], eta))
            weights[i] = np.divide(np.exp(np.divide(bellman_errors[i], eta)), denominator)

        partial = np.zeros((number_of_samples, 1))
        for i in range(0, number_of_samples):
            partial = np.add(partial, np.multiply(weights[i], embedding_vectors[:, i]).reshape((number_of_samples, 1)))

        alpha = np.add(alpha, -partial)

    return alpha


def minimize_dual_for_eta():
    # Constrained opt
    # TODO: implement
    return 0.0


def strictly_bigger(vec1, vec2):
    """

    :param vec1:
    :param vec2:
    :return:
    """
    # TODO: Comment
    booleans = np.greater(vec1 , vec2)
    bigger = True
    for i in range(0, np.shape(booleans)[0]):
        if bigger:
            bigger = booleans[i]
        else:
            break
    return bigger


def update_step(old_policy):

    """ 1. generate roll-outs according to pi_(i-1) """

    samples = generate_episode(ENV_PENDULUM, POLICY_GAUSSIAN)
    number_of_samples = np.shape(samples)[0]

    """ 2. calculate kernel embedding strengths """
    """ Which means prepare the Kernelmatrix so you can evaluate beta(s,a) """
    K_sa = evaluate_state_action_kernel(samples)

    """ 3. minimize kernel-based dual """

    # iterate coordinate-descent (till constraints are sufficiently fulfilled)
    alpha = np.ones((number_of_samples, 1))*0.01
    eta = 0.01
    # convergence threshold
    epsilon = 0.1
    for i in range(0, 10):
    #while (strictly_bigger(alpha, np.ones((number_of_samples, 1))*epsilon)) | (eta > epsilon):

        # fast unconstrained convex optimization on alpha (fixed iterations)
        alpha = minimize_dual_for_alpha(alpha, eta, samples)

        # constrained minimization on eta (fixed iterations)
        for iteration in range(0, 10):
            eta = minimize_dual_for_eta()

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
