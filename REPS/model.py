import gym
import time
import numpy as np


class Model:

    def __init__(self, initial_samples,l_c=2.0, l=2.0, bandwidth_matrix=None):
        self.samples = initial_samples
        self.transition_kernel = None
        self.state_action_kernel = None
        self.embedding_vectors = None
        self.inv_reg_kernel = None
        if bandwidth_matrix is None:
            self.bandwidth_matrix = np.identity(np.shape(self.samples)[0])
        else:
            self.bandwidth_matrix = bandwidth_matrix
        self.transition_regulator = l
        self.state_action_regulator = l_c
        self.setup()

    def setup(self):
        self.evaluate_state_action_kernel()
        self.evaluate_transition_kernel()
        number_of_samples = np.shape(self.samples)[0]
        self.embedding_vectors = np.zeros((number_of_samples, number_of_samples))
        for i in range(0, number_of_samples):
            sample = self.samples[i]
            self.embedding_vectors[:, i] = self.calculate_embedding_vector(sample[0], sample[1])\
                                           .reshape(number_of_samples, )

        self.inv_reg_kernel = np.linalg.inv(np.add(self.transition_kernel, self.transition_regulator * self.bandwidth_matrix))


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
        self.state_action_kernel = np.zeros((number_of_samples, number_of_samples))
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
                self.state_action_kernel[i][j] = state_kval * action_kval
        return None

    def evaluate_transition_kernel(self):
        """
        Evaluated a single kernelized model of the state transition

        :param samples: set of state-action pairs available to calculate the state-transition kernel
                        (n_samples x len(samples))
        :return: Matrix defining the state-transition kernel
                 (n_samples x n_samples)
        """
        number_of_samples = np.shape(self.samples)[0]
        self.transition_kernel = np.zeros((number_of_samples, number_of_samples))
        sample_i = None
        sample_j = None
        for i in range(0, number_of_samples):
            sample_i = self.samples[i]
            for j in range(0, number_of_samples):
                sample_j = self.samples[j]
                self.transition_kernel[i][j] = self.gaussian_kernel(sample_i[0], sample_j[2])
        return None

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

        if not speed_up:
            dif_vec = (np.array(vec1) - np.array(vec2)).astype(np.float32)
            if bandwidth_matrix is None:
                bandwidth_matrix = np.identity(np.shape(dif_vec)[0], np.float32)
            result = np.exp(np.matmul(np.matmul(-np.transpose(dif_vec),
                                                bandwidth_matrix),
                                      dif_vec))
        return result

    def calculate_embedding_vector(self, state, action):
        """
        Precomputation for the bellman-error
        :param state: state for which to evaluate the embedding vector
                      (len(state) x 1)
        :param action: action for which to evaluate the embedding vector
                       (len(state) x 1)
        :return: the embedding vector (s,a) needed for the bellman-error
        """

        start_time = time.clock()
        last_time = start_time

        beta = self.calculate_beta(state, action)

        beta_time = time.clock() - last_time
        last_time = time.clock()

        number_of_samples = np.shape(self.samples)[0]
        transition_vec = np.zeros((number_of_samples, 1))
        for i in range(0, number_of_samples):
            transition_vec[i] = self.gaussian_kernel(self.samples[i][0], state)

        transition_time = time.clock() - last_time
        last_time = time.clock()

        embedding_vec = np.add(np.matmul(self.transition_kernel,
                                         beta),
                               -transition_vec)

        matrix_time = time.clock() - last_time
        full_time = time.clock() - start_time
        # print("--- %s percent --- calculate_beta" % (round(beta_time / full_time, 2)))
        # print("--- %s percent --- transition_vec" % (round(transition_time / full_time, 2)))
        # print("--- %s percent --- matrices" % (round(matrix_time / full_time, 2)))
        return embedding_vec

    def calculate_beta(self, state, action):
        """
        Precomputation for the embedding_vectors
        :param state: state-argument for which to evaluate beta(s,a)
        :param action: action-argument for which to evaluate beta(s,a)
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

        reg_mat = np.multiply(self.state_action_regulator, np.identity(number_of_samples))
        beta = np.matmul(np.linalg.inv(np.add(self.state_action_kernel,
                                             reg_mat)),
                         state_action_vec)

        # print("--- %s percent --- kernel_time" % (round(kernel_time / (time.clock() - start_time), 2)))
        # print("--- %s times --- faster" % (round(kernel_time / improved_time, 2)))
        return beta

