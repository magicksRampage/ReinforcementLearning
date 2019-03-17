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
    state_kval = 0.0
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


def penalty(eta):
    """

    :param eta: positive Lagrange-parameter
    :return: penalty for an eta to close to 0
    """
    return -np.log(eta)


def penalty_derivative(eta):
    """

    :param eta: positive Lagrange-parameter
    :return: derivative of the penalty for an eta to close to 0
    """
    return -1/eta


def evaluate_dual(alpha, eta, samples, transition_kernel=None, state_action_kernel=None, embedding_vectors=None, epsilon=1.0):
    """

    :param alpha:
    :param eta:
    :param samples:
    :param epsilon:
    :param transition_kernel:
    :param state_action_kernel:
    :param embedding_vectors:
    :return:
    """
    # TODO: Comment

    number_of_samples = np.shape(samples)[0]
    if transition_kernel is None:
        # TODO: If transition_kernel is not given recalculate it
        return None
    if state_action_kernel is None:
        # TODO: If state_action_kernel is not given recalculate it
        return None
    if embedding_vectors is None:
        # TODO: If embedding_vectors are not given recalculate them
        return None

    bellman_errors = np.zeros((number_of_samples, 1))
    log_sum = 0.0
    log_regulator = -np.inf

    sample = None
    for i in range(0, number_of_samples):
        sample = samples[i]
        bellman_errors[i] = calculate_bellman_error(sample[3], alpha, embedding_vectors[:, i])

    for i in range(0, number_of_samples):
        if np.divide(bellman_errors[i], eta) > log_regulator:
            log_regulator = np.divide(bellman_errors[i], eta)

    for i in range(0, number_of_samples):
        # going to log space to avoid numerical issues
        log_sum += np.exp(np.divide(bellman_errors[i], eta) - log_regulator)
        # Avoid crash by log(0)
        # TODO: Find a more elegant way to avoid a crash
        if log_sum == 0:
            log_sum = 1.0e-10
    g = eta * epsilon + eta * np.log(log_sum) + penalty(eta)
    return g


def gaussian_kernel(vec1, vec2, bandwidth_matrix=None):
    """

    :param vec1: first argument in array form (scalars will cause problems due to np.transpose())
    :param vec2: second argument in array form (scalars will cause problems due to np.transpose())
    :param bandwidth_matrix: Matrix defining the free bandwidth parameters of a gaussian kernel (must be len(vec1) == len(vec2)
    :return: scalar result of the kernel evaluation
    """
    dif_vec = vec1 - vec2
    if bandwidth_matrix is None:
        bandwidth_matrix = np.identity(np.shape(dif_vec)[0])
    return np.exp(np.matmul(np.matmul(-np.transpose(dif_vec), bandwidth_matrix), dif_vec))


def calculate_beta(state, action, samples, state_action_kernel, l_c=-1.0):
    """

    :param state: state-argument for which to evaluate beta(s,a)
    :param action: action-argument for which to evaluate beta(s,a)
    :param samples: set of state-action pairs available to calculate beta(s,a)
    :param state_action_kernel: the pre-calculated state-action kernel matrix
    :param l_c: regularization coefficient
    :return: beta(s,a)
    """
    number_of_samples = np.shape(samples)[0]
    state_action_vec = np.zeros((number_of_samples, 1))
    sample = None

    for i in range(0, number_of_samples):
        sample = samples[i]
        state_kval = gaussian_kernel(sample[0], state)
        action_kval = gaussian_kernel(np.array([sample[1]]), np.array([action]))
        state_action_vec[i] = state_kval * action_kval

    reg_mat = np.multiply(l_c, np.identity(number_of_samples))
    beta = np.matmul(np.transpose(np.add(state_action_kernel, reg_mat)), state_action_vec)
    return beta


def calculate_embedding_vector(state, action, samples, transition_kernel, state_action_kernel):
    """

    :param state: state-argument for which to evaluate the embedding vector
    :param action: action-argument for which to evaluate the embedding vector
    :param samples: set of state-action pairs available to calculate the embedding vector
    :param transition_kernel: the pre-calculated state-action kernel matrix
    :param state_action_kernel:
    :return: the embedding vector (s,a) needed for the bellman-error
    """

    beta = calculate_beta(state, action, samples, state_action_kernel)

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


def calculate_weights(samples, bellman_errors, eta):
    """

    :param samples:
    :param bellman_errors:
    :param eta:
    :return:
    """
    # TODO: Comment

    number_of_samples = np.shape(samples)[0]
    weights = np.zeros((number_of_samples, 1))

    for i in range(0, number_of_samples):
        log_denominator = 0.0
        log_regulator = 0.0

        # Find the right regulator
        for j in range(i, number_of_samples):
            if np.divide(bellman_errors[j], eta) > log_regulator:
                log_regulator = np.divide(bellman_errors[j], eta)

        for j in range(i, number_of_samples):
            # going to log space to avoid numerical issues
            log_denominator += np.exp(np.divide(bellman_errors[j], eta) - log_regulator)
            # Avoid crash by log(0)
            # TODO: Find a more elegant way to avoid a crash
            if log_denominator == 0:
                log_denominator = 1.0e-10
        weights[i] = np.exp(np.divide(bellman_errors[i], eta) - (np.log(log_denominator) + log_regulator))

    return weights


def minimize_dual(initial_alpha, initial_eta, samples, transition_kernel, state_action_kernel, embedding_vectors, epsilon=1.0):
    """

    :param initial_alpha:
    :param initial_eta:
    :param samples:
    :param  transition_kernel:
    :param state_action_kernel:
    :param embedding_vectors:
    :param epsilon:
    :return:
    """
    # TODO: Comment
    number_of_samples = np.shape(samples)[0]

    old_alpha = initial_alpha
    old_eta = initial_eta
    temp_alpha = None
    temp_eta = None

    temp_dual_value = evaluate_dual(old_alpha, old_eta, samples, transition_kernel, state_action_kernel, embedding_vectors)
    old_dual_value = np.inf

    improvement = old_dual_value - temp_dual_value
    while improvement > 0.01:
        old_dual_value = temp_dual_value

        # fast unconstrained convex optimization on alpha (fixed iterations)
        temp_alpha = minimize_dual_for_alpha(old_alpha, old_eta, samples, embedding_vectors)

        # constrained minimization on eta (fixed iterations)
        temp_eta = minimize_dual_for_eta(temp_alpha, old_eta, samples, embedding_vectors, epsilon)

        temp_dual_value = evaluate_dual(temp_alpha, temp_eta, samples, transition_kernel, state_action_kernel, embedding_vectors)
        improvement = old_dual_value - temp_dual_value
        if improvement > 0.01:
            old_alpha = temp_alpha
            old_eta = temp_eta
            old_dual_value = temp_dual_value
        else:
            print("converged")

        print(improvement, temp_eta, temp_dual_value)

    return [old_alpha, old_eta]


def minimize_dual_for_alpha(initial_alpha, eta, samples, embedding_vectors, number_of_iterations=10, step_size=0.001):
    """

    :param initial_alpha:
    :param eta:
    :param samples:
    :param embedding_vectors:
    :param number_of_iterations:
    :param step_size:
    :return:
    """
    # TODO: Comment

    number_of_samples = np.shape(samples)[0]
    alpha = initial_alpha
    # Initialize values but once
    bellman_errors = np.zeros((number_of_samples, 1))
    temp = np.zeros((number_of_samples, 1))
    weights = np.zeros((number_of_samples, 1))
    partial = np.zeros((number_of_samples, 1))

    for descent in range(0, number_of_iterations):
        sample = None
        for i in range(0, number_of_samples):
            sample = samples[i]
            bellman_errors[i] = calculate_bellman_error(sample[3], alpha, embedding_vectors[:, i])

        weights = calculate_weights(samples, bellman_errors, eta)

        partial = np.zeros((number_of_samples, 1))
        # Ignore the penalty for eta as it is constant
        for i in range(0, number_of_samples):
            partial = np.add(partial, np.multiply(weights[i], embedding_vectors[:, i]).reshape((number_of_samples, 1)))

        # TODO: Refine step-length (hessian?)
        alpha = np.add(alpha, -partial * step_size)

    return alpha


def minimize_dual_for_eta(alpha, initial_eta, samples, embedding_vectors, epsilon, number_of_iterations=10, step_size=0.001):
    """

    :param alpha:
    :param initial_eta:
    :param samples:
    :param embedding_vectors:
    :param epsilon:
    :param number_of_iterations:
    :param step_size:
    :return:
    """
    # TODO: Comment

    number_of_samples = np.shape(samples)[0]
    eta = initial_eta
    # Initialize values but once
    bellman_errors = np.zeros((number_of_samples, 1))
    weights = np.zeros((number_of_samples, 1))

    sample = None
    for i in range(0, number_of_samples):
        sample = samples[i]
        bellman_errors[i] = calculate_bellman_error(sample[3], alpha, embedding_vectors[:, i])

    for descent in range(0, number_of_iterations):
        weights = calculate_weights(samples, bellman_errors, eta)

        partial = 0.0
        weight_sum = 0.0

        for i in range(0, number_of_samples):
            weight_sum += weights[i]*bellman_errors[i]

        log_sum = 0.0
        log_regulator = -np.inf

        # Find the right regulator
        for i in range(0, number_of_samples):
            if np.divide(bellman_errors[i], eta) > log_regulator:
                log_regulator = np.divide(bellman_errors[i], eta)

        for i in range(0, number_of_samples):
            # going to log space to avoid numerical issues
            log_sum += np.exp(np.divide(bellman_errors[i], eta) - log_regulator)
            # Avoid crash by log(0)
            # TODO: Find a more elegant way to avoid a crash
            if log_sum == 0:
                log_sum = 1.0e-10

        partial = -(1/eta)*weight_sum + epsilon + np.log(log_sum) + log_regulator - np.log(number_of_samples) + penalty_derivative(eta)
        # TODO: Refine step-length (hessian?)
        eta = eta - partial*step_size
        # Clamp eta to be non zero
        # TODO: Find a real way to handle the instability
        if eta < 0.01:
            eta = 0.01

    return eta


def update_step(old_policy):
    """ 1. generate roll-outs according to pi_(i-1) """

    samples = generate_episode(ENV_PENDULUM, POLICY_GAUSSIAN)
    number_of_samples = np.shape(samples)[0]

    """ 2. calculate kernel embedding strengths """
    """ 
        Which means prepare the embedding vectors [K_s*beta(s_i,a_i) - k_s(s_i)]
        The state_action_kernel binds states to corresponding actions 
        The transition_kernel binds states to following states (independent of actions) 
    """
    state_action_kernel = evaluate_state_action_kernel(samples)
    transition_kernel = evaluate_state_transition_kernel(samples)

    embedding_vectors = np.zeros((number_of_samples, number_of_samples))
    for i in range(0, number_of_samples):
        sample = samples[i]
        embedding_vectors[:, i] = calculate_embedding_vector(sample[0], sample[1], samples, transition_kernel,
                                                             state_action_kernel).reshape(number_of_samples, )

    """ 3. minimize kernel-based dual """

    # iterate coordinate-descent (till constraints are sufficiently fulfilled)
    alpha = np.ones((number_of_samples, 1)) * 0.01
    eta = 1
    [alpha, eta] = minimize_dual(alpha, eta, samples, transition_kernel, state_action_kernel, embedding_vectors)

    """ 4. calculate kernel-based Bellman errors """

    bellman_errors = np.zeros((number_of_samples, 1))
    for i in range(0, number_of_samples):
        sample = samples[i]
        bellman_errors[i] = calculate_bellman_error(sample[3], alpha, embedding_vectors[:, i])

    """ 5. calculate the sample weights """

    weights = np.zeros((number_of_samples, 1))
    weights = calculate_weights(samples, bellman_errors, eta)

    """ 6. fit a generalizing non-parametric policy"""

    # hyper-parameter
    print("Please implement the policy fitting :)")
    new_policy = None

    return new_policy


def main():
    converged = False
    # Parameters for a gaussian policy (mu, sigma_sq)
    policy = (0, 1)
    while not converged:
        policy = update_step(policy)
        # TODO: define conversion target
        converged = True


main()
