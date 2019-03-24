import gym
import time
import numpy as np


class Agent:

    def __init__(self, env_name):
        self.environment = env_name
        self.samples = self.generate_initial_episode()
        self.policy = None

    def main(self):
        converged = False
        for i in range(0, 50):
            # TODO: define conversion target
            self.policy = self.update_step()

    def generate_initial_episode(self):
        """
        Interacts a with an environment to produce the initial trajectory.
        If they aren't a random policy is employed.

        :param env_name: Global name of the environment
        :return: an array containing the sequence of samples generated in this episode
                 (n_samples x len(samples))
        """
        env = gym.make(self.environment)
        env.seed(0)
        prev_obs = env.reset()
        new_samples = ()
        low_action = env.action_space.low[0]
        high_action = env.action_space.high[0]
        action = 0

        for i in range(0, 100):
            action = self.random_policy(low_action, high_action)

            obs, reward, done, info = env.step((action,))
            new_samples += ((prev_obs, action, obs, reward),)
            prev_obs = obs
            env.render()

        env.close()
        return new_samples

    def update_step(self, memory_size=500):
        """
        Optimize the policy according to the following steps:
        1. generate roll-outs according to pi_(i-1)
        2. calculate kernel embedding strengths
        3. minimize kernel-based dual
        4. calculate kernel-based Bellman errors
        5. calculate the sample weights
        6. fit a generalizing non-parametric policy
        This function contains many lines of profiling code!
        :param memory_size: the limit on the agents memory limit
        :return: the policy_parameters of the next policy
        """
        last_time = time.clock()

        """ 1. generate roll-outs according to pi_(i-1) """
        temp_samples = None
        old_samples = None
        if self.policy is None:
            temp_samples = self.samples
        else:
            old_policy_parameters = self.policy
            # (alpha, eta, samples, transition_kernel, embedding_vectors)
            old_alpha = old_policy_parameters[0]
            old_eta = old_policy_parameters[1]
            old_transition_kernel = old_policy_parameters[2]
            old_embedding_vectors = old_policy_parameters[3]
            temp_samples = self.generate_episode(old_alpha,
                                                 old_eta,
                                                 old_transition_kernel,
                                                 old_embedding_vectors)

        if self.policy is not None:
            temp_samples = temp_samples + self.samples
            # TODO: Generalize
            if np.shape(temp_samples)[0] > memory_size:
                temp_samples = temp_samples[:memory_size]

        self.samples = temp_samples
        print(np.shape(temp_samples)[0])

        number_of_samples = np.shape(self.samples)[0]

        print("Step 1 --- %s seconds ---" % (round(time.clock() - last_time, 2)))
        last_time = time.clock()

        """ 2. calculate kernel embedding strengths """
        """ 
            Which means prepare the embedding vectors [K_s*beta(s_i,a_i) - k_s(s_i)]
            The state_action_kernel binds states to corresponding actions ==> pi(a|s)
            The transition_kernel binds states to following states ==> p_pi(s,a) 
        """
        start_step_2_time = time.clock()
        step_2_time = start_step_2_time

        state_action_kernel = self.evaluate_state_action_kernel()

        state_action_time = step_2_time - time.clock()
        step_2_time = time.clock()

        transition_kernel = self.evaluate_state_transition_kernel()

        transition_time = step_2_time - time.clock()
        step_2_time = time.clock()

        embedding_vectors = np.zeros((number_of_samples, number_of_samples))
        for i in range(0, number_of_samples):
            sample = self.samples[i]
            embedding_vectors[:, i] = self.calculate_embedding_vector(sample[0],
                                                                 sample[1],
                                                                 transition_kernel,
                                                                 state_action_kernel).reshape(number_of_samples, )

        embedding_time = step_2_time - time.clock()
        step_2_time = time.clock()
        full_step_2_time = start_step_2_time - time.clock()
        # print("--- %s percent --- state_action_kernel" % (state_action_time / full_step_2_time))
        # print("--- %s percent --- transition_kernel" % (transition_time / full_step_2_time))
        # print("--- %s percent --- embedding_vectors" % (embedding_time / full_step_2_time))

        print("Step 2 --- %s seconds ---" % (round(time.clock() - last_time, 2)))
        last_time = time.clock()

        """ 3. minimize kernel-based dual """

        # iterate coordinate-descent (till constraints are sufficiently fulfilled)
        alpha = np.ones((number_of_samples, 1)) * -0.1
        eta = 1
        [alpha, eta] = self.minimize_dual(alpha,
                                     eta,
                                     transition_kernel,
                                     state_action_kernel,
                                     embedding_vectors)

        print("Step 3 --- %s seconds ---" % (round(time.clock() - last_time, 2)))
        last_time = time.clock()

        """ 4. calculate kernel-based Bellman errors """

        bellman_errors = np.zeros((number_of_samples, 1))
        for i in range(0, number_of_samples):
            sample = self.samples[i]
            bellman_errors[i] = self.calculate_bellman_error(sample[3],
                                                        alpha,
                                                        embedding_vectors[:, i])

        print("Step 4 --- %s seconds ---" % (round(time.clock() - last_time, 2)))
        last_time = time.clock()

        """ 5. calculate the sample weights """

        weights = np.zeros((number_of_samples, 1))
        weights = self.calculate_weights(eta, bellman_errors)

        print("Step 5 --- %s seconds ---" % (round(time.clock() - last_time, 2)))
        last_time = time.clock()

        """ 6. fit a generalizing non-parametric policy """

        space_low = None
        space_high = None
        new_policy_parameters = (alpha,
                                 eta,
                                 transition_kernel,
                                 embedding_vectors)

        print("Step 6 --- %s seconds ---" % (round(time.clock() - last_time, 2)))
        last_time = time.clock()

        return new_policy_parameters

    def generate_episode(self, alpha, eta, transition_kernel,
                         embedding_vectors, bandwidth_matrix=None, l=1.0):
        """
        Interacts a with an environment to produce a trajectory.
        All parameters need to be set here.
        If they are one can set the hyperparameters as well.
        If they aren't a random policy is employed.

        :param env_name: Global name of the environment
        :param alpha: a vector-lagrangian-parameter
                      (n_samples x 1)
        :param eta: a scalar lagrangian-parameter
        :param old_samples: the previous samples in the "memory" of the agent
                            (n_samples x len(samples))
        :param transition_kernel: the kernelized model of the transition-distribution p(s, s')
                                  (n_samples x n_samples)
        :param embedding_vectors: storage vector of precomputation
                                  (n_samples x 1)
        :param bandwidth_matrix: regularization hyperparameter
                                 (n_samples x n_samples)
        :return: an array containing the sequence of samples generated in this episode
                 (n_samples x len(samples))
        """
        env = gym.make(self.environment)
        env.seed(0)
        prev_obs = env.reset()
        new_samples = ()
        low_action = env.action_space.low[0]
        high_action = env.action_space.high[0]
        action = 0

        number_of_old_samples = np.shape(self.samples)[0]
        bandwidth_matrix = np.identity(number_of_old_samples)
        inv_reg_kernel = np.linalg.inv(np.add(transition_kernel, l * bandwidth_matrix))

        # weights are an estimate of the Advantage-function
        bellman_errors = np.zeros((number_of_old_samples, 1))
        for i in range(0, number_of_old_samples):
            sample = self.samples[i]
            bellman_errors[i] = self.calculate_bellman_error(sample[3],
                                                                alpha,
                                                                embedding_vectors[:, i])

        advantages = self.calculate_weights(eta, bellman_errors)

        for i in range(0, 100):
            action = self.gaussian_policy(prev_obs,
                                          low_action,
                                          high_action,
                                          alpha,
                                          eta,
                                          inv_reg_kernel,
                                          advantages,
                                          l)[0]

            obs, reward, done, info = env.step((action,))
            new_samples += ((prev_obs, action, obs, reward),)
            prev_obs = obs
            env.render()
            time.sleep(0.01)

        env.close()
        return new_samples

    def random_policy(self, space_low, space_high):
        """
        Defines a random policy over a one-dimensional action-space

        :param space_low:  low end of the action space
        :param space_high:   high end of the action space
        :return: returns a sample from a equiprobable distribution over the action space
        """

        space_range = space_high - space_low
        sample = space_range * np.random.random() + space_low
        return sample

    def gaussian_policy(self, state, space_low, space_high, alpha, eta, inverted_kernel, advantages, l):
        """
        Defines a kernelized policy by returning a n-dimensional action for a state given the parameters
        (the parameters contain pre-computations of the policy-evalutation)

        :param state: the state for which an action is to be returned
        :param space_low: a vector containing the minimal value of all action-dimensions
                          (n_samples x 1)
        :param space_high: a vector containing the maximal value of all action-dimensions
                           (n_samples x 1)
        :param alpha: a vector-lagrangian-parameter
                      (n_samples x 1)
        :param eta: a scalar lagrangian-parameter
        :param samples: the samples in the "memory" of the agent
                        (n_samples x len(samples))
        :param inverted_kernel: the regularized and then inverted transition_kernel
                                (n_samples x n_samples)
        :param advantages: a vector containing the advantage-values for all samples
                           (n_samples x 1)
        :param l: Regularization-hyperparameter
        :return: the action defined by the policy given the state
                 (len(action) x 1)
        """
        # TODO: Hyperparameter D and l

        number_of_samples = np.shape(self.samples)[0]

        transition_vec = np.zeros((number_of_samples, 1))
        sample = None
        for i in range(0, number_of_samples):
            sample = self.samples[i]
            transition_vec[i] = self.gaussian_kernel(sample[0], state)

        mean = np.matmul(np.transpose(transition_vec),
                         np.matmul(inverted_kernel,
                                   advantages))[0]

        variance = self.gaussian_kernel(state, state) + l - np.matmul(np.transpose(transition_vec),
                                                                 np.matmul(inverted_kernel,
                                                                           transition_vec))[0]
        if variance < 0:
            print("Pls check variance!")
        standard_deviation = np.sqrt(variance)
        action = self.sample_gaussian(space_low,
                                 space_high,
                                 mean,
                                 standard_deviation)

        return action

    def sample_gaussian(self, space_low, space_high, mean, standard_deviation):
        """
        Samples a simple gaussian distribuition given its parameter.
        Values that exceed the bounds of the space are clipped

        :param space_low: low end of the action space
                          (len(action) x 1)
        :param space_high: high end of the action space
                           (len(action) x 1)
        :param mean: mean of the gaussian
                     (len(action) x 1)
        :param standard_deviation: standard deviation of the gaussian
                                   (len(action) x 1)
        :return: a sample from a gaussian distribution over the action space
                 (len(action) x 1)
        """

        space_range = space_high - space_low
        sample = np.random.normal(mean, standard_deviation)
        return np.clip(sample,
                       space_low,
                       space_high)

    def evaluate_state_action_kernel(self):
        """
        Evaluates a single kernelized model of which action actually occur in the for given states
        (Differs from the definition of a policy if its outcome is stochastic)

        :param samples: set of state-action pairs available to calculate the state-action kernel
                        (n_samples x len(samples))
        :return: Matrix defining the state-action kernel
                 (n_samples x n_samples)
        """
        number_of_samples = np.shape(self.samples)[0]
        state_action_kernel = np.zeros((number_of_samples, number_of_samples))
        sample_i = None
        sample_j = None
        state_kval = 0.0
        action_kval = 0.0
        for i in range(0, number_of_samples):
            sample_i = self.samples[i]
            for j in range(0, number_of_samples):
                sample_j = self.samples[j]
                state_kval = self.gaussian_kernel(sample_i[0], sample_j[0])
                action_kval = self.gaussian_kernel(np.array([sample_i[1]]), np.array([sample_j[1]]))
                state_action_kernel[i][j] = state_kval * action_kval
        return state_action_kernel

    def evaluate_state_transition_kernel(self):
        """
        Evaluated a single kernelized model of the state transition

        :param samples: set of state-action pairs available to calculate the state-transition kernel
                        (n_samples x len(samples))
        :return: Matrix defining the state-transition kernel
                 (n_samples x n_samples)
        """
        number_of_samples = np.shape(self.samples)[0]
        state_transition_kernel = np.zeros((number_of_samples, number_of_samples))
        sample_i = None
        sample_j = None
        for i in range(0, number_of_samples):
            sample_i = self.samples[i]
            for j in range(0, number_of_samples):
                sample_j = self.samples[j]
                state_transition_kernel[i][j] = self.gaussian_kernel(sample_i[0], sample_j[2])
        return state_transition_kernel

    def gaussian_kernel(self, vec1, vec2, speed_up=False, bandwidth_matrix=None):
        """
        Defines the value of the gaussian_kernel between two vectors
        :param vec1: first argument in array form (scalars will cause problems due to np.transpose())
        :param vec2: second argument in array form (scalars will cause problems due to np.transpose())
        :param speed_up: Debugging boolean
        :param bandwidth_matrix: Matrix defining the free bandwidth parameters of a gaussian kernel
                                (len(vec1) x len(vec2))
        :return: scalar result of the kernel evaluation
        """
        # TODO: Hyperparameter D
        if speed_up:
            speed_up = False
            # TODO: Bring to calculation to GPU

        elif not speed_up:
            dif_vec = (vec1 - vec2).astype(np.float32)
            if bandwidth_matrix is None:
                bandwidth_matrix = np.identity(np.shape(dif_vec)[0], np.float32)
            result = np.exp(np.matmul(np.matmul(-np.transpose(dif_vec),
                                                bandwidth_matrix),
                                      dif_vec))
        return result

    def calculate_embedding_vector(self, state, action, transition_kernel, state_action_kernel):
        """
        Precomputation for the bellman-error
        :param state: state for which to evaluate the embedding vector
                      (len(state) x 1)
        :param action: action for which to evaluate the embedding vector
                       (len(state) x 1)
        :param samples: set of state-action pairs available to calculate the state-transition kernel
                        (n_samples x len(samples))
        :param transition_kernel: the kernelized model of the transition-distribution p(s, s')
                                  (n_samples x n_samples)

        :param state_action_kernel:the kernelized model of the action-distribution p(a|s)
                                  (n_samples x n_samples)
        :return: the embedding vector (s,a) needed for the bellman-error
        """

        start_time = time.clock()
        last_time = start_time

        beta = self.calculate_beta(state,
                              action,
                              state_action_kernel)

        beta_time = time.clock() - last_time
        last_time = time.clock()

        number_of_samples = np.shape(self.samples)[0]
        transition_vec = np.zeros((number_of_samples, 1))
        sample = None
        for i in range(0, number_of_samples):
            transition_vec[i] = self.gaussian_kernel(self.samples[i][0], state)

        transition_time = time.clock() - last_time
        last_time = time.clock()

        embedding_vec = np.add(np.matmul(transition_kernel,
                                         beta),
                               -transition_vec)

        matrix_time = time.clock() - last_time
        full_time = time.clock() - start_time
        # print("--- %s percent --- calculate_beta" % (round(beta_time / full_time, 2)))
        # print("--- %s percent --- transition_vec" % (round(transition_time / full_time, 2)))
        # print("--- %s percent --- matrices" % (round(matrix_time / full_time, 2)))
        return embedding_vec

    def calculate_beta(self, state, action, state_action_kernel, l_c=-1.0):
        """
        Precomputation for the embedding_vectors
        :param state: state-argument for which to evaluate beta(s,a)
        :param action: action-argument for which to evaluate beta(s,a)
        :param samples: the samples in the "memory" of the agent
                        (n_samples x len(samples))
        :param state_action_kernel:the kernelized model of the action-distribution p(a|s)
                                  (n_samples x n_samples)
        :param l_c: regularization coefficient
        :return: beta(s,a)
        """
        start_time = time.clock()
        kernel_time = 0.0
        improved_time = 0.0

        number_of_samples = np.shape(self.samples)[0]
        state_action_vec = np.zeros((number_of_samples, 1))
        sample = None

        for i in range(0, number_of_samples):
            sample = self.samples[i]

            before_kernel = time.clock()
            state_kval = self.gaussian_kernel(sample[0], state)
            action_kval = self.gaussian_kernel(np.array([sample[1]]), np.array([action]))
            kernel_time += time.clock() - before_kernel

            """
            ---Debugging code---
            before_kernel = time.clock()
            state_kval = gaussian_kernel(sample[0], state, True)
            action_kval = gaussian_kernel(np.array([sample[1]]), np.array([action]), True)
            improved_time += time.clock() - before_kernel
            """

            state_action_vec[i] = state_kval * action_kval

        reg_mat = np.multiply(l_c, np.identity(number_of_samples))
        beta = np.matmul(np.transpose(np.add(state_action_kernel,
                                             reg_mat)),
                         state_action_vec)

        # print("--- %s percent --- kernel_time" % (round(kernel_time / (time.clock() - start_time), 2)))
        # print("--- %s times --- faster" % (round(kernel_time / improved_time, 2)))
        return beta

    def minimize_dual(self, initial_alpha, initial_eta, transition_kernel, state_action_kernel, embedding_vectors,
                      epsilon=0.5):
        """
        Minimize the lagrangian dual in a coordinate-descent-like approach.
        Minimize alpha and eta seperatly for aw fixed amount of iteration.
        Repeat till conversion.
        alpha is minimized unconstrained.
        eta is constrained by eta > 0 (eta ~= 0 breaks the numerics)
        :param initial_alpha: a vector-lagrangian-parameter
                              (n_samples x 1)
        :param initial_eta: a scalar lagrangian-parameter
        :param samples: the samples in the "memory" of the agent
                        (n_samples x len(samples))
        :param transition_kernel: the kernelized model of the transition-distribution p(s,s')
                                  (n_samples x n_samples)
        :param state_action_kernel:the kernelized model of the action-distribution p(a|s)
                                  (n_samples x n_samples)
        :param embedding_vectors: storage vector of precomputation for the bellman errors
                                  (len(embedding_vector) x n_samples)
        :param epsilon: hyperparameter defining the exploration vs exploitation trade-off
                        [0, 1]
        :return: the new alpha and eta which "minimize" the lagrangian dual
        """
        start_time = time.clock()
        full_eval_time = 0.0
        full_alpha_time = 0.0
        full_eta_time = 0.0
        last_time = start_time

        number_of_samples = np.shape(self.samples)[0]

        old_alpha = initial_alpha
        old_eta = initial_eta
        temp_alpha = None
        temp_eta = None

        temp_dual_value = self.evaluate_dual(old_alpha,
                                        old_eta,
                                        transition_kernel,
                                        state_action_kernel,
                                        embedding_vectors)
        old_dual_value = np.inf

        eval_time = time.clock() - last_time
        full_eval_time += eval_time
        last_time = time.clock()

        improvement = old_dual_value - temp_dual_value
        number_of_improvements = 0
        while (improvement > 0.0) & (number_of_improvements < 20):
            old_dual_value = temp_dual_value

            # fast unconstrained convex optimization on alpha (fixed iterations)
            temp_alpha = self.minimize_dual_for_alpha(old_alpha,
                                                 old_eta,
                                                 embedding_vectors,
                                                 3)

            alpha_time = time.clock() - last_time
            last_time = time.clock()

            # constrained minimization on eta (fixed iterations)
            temp_eta = self.minimize_dual_for_eta(temp_alpha,
                                             old_eta,
                                             embedding_vectors,
                                             epsilon,
                                             3)

            eta_time = time.clock() - last_time
            last_time = time.clock()

            temp_dual_value = self.evaluate_dual(temp_alpha,
                                            temp_eta,
                                            transition_kernel,
                                            state_action_kernel,
                                            embedding_vectors)

            improvement = old_dual_value - temp_dual_value

            eval_time = time.clock() - last_time
            last_time = time.clock()

            full_eval_time += eval_time
            full_alpha_time += alpha_time
            full_eta_time += eta_time

            print(improvement, temp_eta, temp_alpha[0])
            number_of_improvements += 1
            if improvement > 0.0:
                old_alpha = temp_alpha
                old_eta = temp_eta
                old_dual_value = temp_dual_value
            else:
                print("converged")


            full_time = time.clock() - start_time
            # print("--- %s percent --- eval_dual " % (round(full_eval_time / full_time, 2)))
            # print("--- %s percent --- alpha " % (round(full_alpha_time / full_time,2)))
            # print("--- %s percent --- eta " % (round(full_eta_time / full_time,2)))

        return [old_alpha, old_eta]

    def evaluate_dual(self, alpha, eta, transition_kernel=None, state_action_kernel=None, embedding_vectors=None,
                      epsilon=1.0):
        """
        Evaluates the lagrangian dual
        :param alpha: a vector-lagrangian-parameter
                      (n_samples x 1)
        :param eta: a scalar lagrangian-parameter
        :param samples: the samples in the "memory" of the agent
                        (n_samples x len(samples))
        :param transition_kernel: the kernelized model of the transition-distribution p(s,s')
                                  (n_samples x n_samples)
        :param state_action_kernel:the kernelized model of the action-distribution p(a|s)
                                  (n_samples x n_samples)
        :param embedding_vectors: storage vector of precomputation for the bellman errors
                                  (len(embedding_vector) x n_samples)
        :param epsilon: hyperparameter defining the exploration vs exploitation trade-off
                        [0, 1]
        :return: the value of the dual for the given lagrangian multiplier and samples
        """
        # TODO: Check Comment for embedding_vectors dimensions
        number_of_samples = np.shape(self.samples)[0]
        # These might be cases we'll never need
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
            sample = self.samples[i]
            bellman_errors[i] = self.calculate_bellman_error(sample[3],
                                                        alpha,
                                                        embedding_vectors[:, i])

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
        g = eta * (epsilon + eta) * (np.log(log_sum) + self.penalty(eta))
        return g

    def calculate_bellman_error(self, reward, alpha, embedding_vector):
        """
        Calculates a bellman_error given lagrangian_multipliers and an embedding
        :param reward: instant reward for a sample
        :param alpha: a vector-lagrangian-parameter
                      (n_samples x 1)
        :param embedding_vector: precomputation for the bellman errors
                                 (n_samples x 1)
        :return: bellman_error for a given sample
        """
        error = reward + np.matmul(np.transpose(alpha), embedding_vector)
        return error

    def penalty(self, eta):
        """
        ---Deprecated--- Defines the penalty term on the lagrangian for eta approaching 0
        :param eta: positive Lagrange-parameter
        :return: penalty for an eta to close to 0
        """
        return 0.0
        # return -np.log(eta)

    def penalty_derivative(self, eta):
        """
        ---Deprecated--- Defines the "penalty term"-derivative on the lagrangian for eta approaching 0
        :param eta: positive Lagrange-parameter
        :return: derivative of the penalty for an eta to close to 0
        """
        return 0.0
        # return -1/eta

    def minimize_dual_for_alpha(self, initial_alpha, eta, embedding_vectors, number_of_iterations=10):
        """
        Minimize the lagrangian dual in "direction" of alpha
        :param initial_alpha: a vector-lagrangian-parameter
                              (n_samples x 1)
        :param eta: a scalar lagrangian-parameter
        :param samples: the samples in the "memory" of the agent
                        (n_samples x len(samples))
        :param embedding_vectors: storage vector of precomputation for the bellman errors
                                  (len(embedding_vector) x n_samples)
        :param number_of_iterations: number of iterations to optimize alpha
        :return: the new value for alpha
                 (n_samples x 1)
        """

        number_of_samples = np.shape(self.samples)[0]
        alpha = initial_alpha
        # Initialize values but once
        bellman_errors = np.zeros((number_of_samples, 1))
        temp = np.zeros((number_of_samples, 1))
        weights = np.zeros((number_of_samples, 1))
        partial = np.zeros((number_of_samples, 1))
        # Trying to improve convergance

        for descent in range(0, number_of_iterations):
            old_alpha = alpha
            sample = None
            for i in range(0, number_of_samples):
                sample = self.samples[i]
                bellman_errors[i] = self.calculate_bellman_error(sample[3],
                                                            alpha,
                                                            embedding_vectors[:, i])

            weights = self.calculate_weights(eta, bellman_errors)

            partial = np.zeros((number_of_samples, 1))
            # Ignore the penalty for eta as it is constant
            for i in range(0, number_of_samples):
                partial = np.add(partial,
                                 np.multiply(weights[i],
                                             embedding_vectors[:, i]).reshape((number_of_samples, 1)))

            hessian = self.calculate_hessian(eta,
                                        embedding_vectors,
                                        bellman_errors,
                                        weights)

            # TODO: Improve convergence
            # try averaging the step and a step-target in log space
            if hessian == 0.0:
                hessian = 1.0e-10
            step_target = 0.5
            step_length = np.exp(1/2 * (np.log(step_target) - np.log(hessian)))
            alpha = np.add(alpha,
                           -partial * step_length)

        return alpha

    def minimize_dual_for_eta(self, alpha, initial_eta, embedding_vectors, epsilon, number_of_iterations=10):
        """
        Minimize the lagrangian dual in "direction" of eta.
        Be aware that an eta sufficiently close to 0 breaks the numerics! : exp(const / eta) is evaluated in the program
        :param alpha: a vector-lagrangian-parameter
                              (n_samples x 1)
        :param initial_eta: a scalar lagrangian-parameter
        :param samples: the samples in the "memory" of the agent
                        (n_samples x len(samples))
        :param embedding_vectors: storage vector of precomputation for the bellman errors
                                  (len(embedding_vector) x n_samples)
        :param epsilon: hyperparameter defining the exploration vs exploitation trade-off
                        [0, 1]
        :param number_of_iterations: number of iterations to optimize eta
        :return: the new value of eta
        """
        number_of_samples = np.shape(self.samples)[0]
        eta = initial_eta
        # Initialize values but once
        bellman_errors = np.zeros((number_of_samples, 1))
        weights = np.zeros((number_of_samples, 1))

        sample = None
        for i in range(0, number_of_samples):
            sample = self.samples[i]
            bellman_errors[i] = self.calculate_bellman_error(sample[3],
                                                        alpha,
                                                        embedding_vectors[:, i])

        for descent in range(0, number_of_iterations):
            weights = self.calculate_weights(eta, bellman_errors)

            partial = 0.0
            weight_sum = 0.0

            for i in range(0, number_of_samples):
                weight_sum += weights[i] * bellman_errors[i]

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
                if log_sum == 0.0:
                    log_sum = 1.0e-10

            partial = -(1 / eta) * weight_sum + epsilon + np.log(log_sum) + log_regulator - np.log(
                number_of_samples) + self.penalty_derivative(eta)
            hessian = self.calculate_hessian(eta,
                                        embedding_vectors,
                                        bellman_errors,
                                        weights)
            # TODO: Improve COnvergence
            step_target = eta * 0.5
            if hessian == 0.0:
                hessian = 1.0e-10
            step_length = np.exp(1/2 * (np.log(step_target) - np.log(hessian)))
            eta = eta - partial * step_length
            # Clamp eta to be non zero
            # TODO: Could we throw a error here without using try(), catch() everywhere?
            if eta < 1.0e-3:
                eta = 1.0e-3

        return eta

    def calculate_weights(self, eta, bellman_errors):
        """
        Calculate the weights for all samples
        :param eta: a scalar lagrangian-parameter
        :param samples: the samples in the "memory" of the agent
                        (n_samples x len(samples))
        :param bellman_errors: the bellman_error for each samples
                               (n_samples x 1)
        :return: the weight for each samples
                 (n_samples x 1)
        """
        # TODO: Put eta in front (consistency)

        number_of_samples = np.shape(self.samples)[0]
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

    def calculate_hessian(self, eta, embedding_vectors, bellman_errors, weights):
        """
        Calculate the second derivative for the lagrangian (which turns out to be scalar)
        :param embedding_vectors: storage vector of precomputation for the bellman errors
                                  (len(embedding_vector) x n_samples)
        :param bellman_errors: the bellman_error for each samples
                               (n_samples x 1)
        :param weights: the weight for each samples
                        (n_samples x 1)
        :return: the scalar value for the hessian
        """
        # Inference of number_of_samples. Each sample has a bellman_error
        number_of_samples = np.shape(bellman_errors)[0]

        # number_of_samples x len(embedding_vector + 1)
        u_vectors = np.zeros((number_of_samples, number_of_samples + 1))
        for i in range(0, number_of_samples):
            u_vectors[i, :] = np.append(embedding_vectors[:, i], bellman_errors[i])

        weighted_u_vectors = np.zeros((number_of_samples, number_of_samples + 1))
        hessian = 0.0
        for i in range(0, number_of_samples):
            for j in range(i, number_of_samples):
                weighted_u_vectors[j, :] += u_vectors[j, :] * weights[j]
            hessian += (1 / eta) * weights[i] * np.dot(weighted_u_vectors[i], weighted_u_vectors[i])

        return hessian

