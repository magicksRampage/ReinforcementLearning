import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn


class Actor:

    def __init__(self, min_action, max_action, samples, q_critic, v_critic, old_actor):
        self.min_action = min_action
        self.max_action = max_action
        self.samples = samples
        self.q_critic = q_critic
        self.v_critic = v_critic
        # The old actor will be none in the first run as we explore randomly
        self.old_actor = old_actor
        self.weights = None
        self._calculate_weights()
        # Parameters of the multivariate Gaussian
        self.mean = None
        self.var = None
        self._fit()

    def _calculate_weights(self):
        states = self.samples[0]
        actions = self.samples[1]
        number_of_samples = np.shape(states)[0]
        vc = self.v_critic
        qc = self.q_critic
        weights = np.zeros((number_of_samples, 1))
        # Numeric Regulation
        for i in range(0, number_of_samples):
            weights[i] = np.exp((qc.estimate_q(states[i], actions[i])
                                 - vc.estimate_v(states[i]))
                                / vc.eta)
            """
            if np.isinf(weights[i]):
                print("Actor-Weights broken")
            """
        self.weights = weights

    def _fit(self):
        # TODO: Use a scalar variance instead of a  full covariance-matrix
        len_state = self.samples[0][0].size
        initial_parameters = np.zeros(len_state + 1)
        initial_parameters[-1] = 1
        res = minimize(self._calc_kl_distance,
                       initial_parameters,
                       jac=False,
                       method='SLSQP',
                       options={'ftol': 1e-6, 'disp': True})
        print(res)
        self.parameter = res.x

    def _calc_kl_distance(self, parameters):
        states = self.samples[0]
        number_of_samples = np.shape(states)[0]
        len_state = states[0].size
        mean = parameters[0:len_state]
        covariance = parameters[-1]
        kl_distance = 0.0
        for i in range(0, number_of_samples):
            kl_distance += (1 / number_of_samples) * self.weights[i]\
                            * mvn.logpdf(states[i], mean, covariance)
            """
            if np.isinf(kl_distance):
                print("KL-Distance broken")
            """
        return kl_distance

    def act(self):
        # TODO: Implement
        return None
