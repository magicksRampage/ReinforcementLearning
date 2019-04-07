"""
Simple class determining and defining a policy.

Functions
---------
- act : returns an action given a state
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn


class Actor:

    def __init__(self, samples, q_critic, v_critic, old_actor):
        """

        :param samples: The observations of the last roll-out containing (states, actions, following_states, rewards)
        :param q_critic: The object estimating the Q-Function
        :param v_critic: The object estimating the V-Function
        :param old_actor: The object defining the previous policy
        """
        self.samples = samples
        self.q_critic = q_critic
        self.v_critic = v_critic
        # The old actor will be none in the first run so we explore randomly
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
            # TODO: Ensure stability
            # !!! Most sensible point for numerics !!!
            weights[i] = np.exp((qc.estimate_q(states[i], actions[i])
                                 - vc.estimate_v(states[i]))
                                / vc.eta)
        self.weights = weights

    def _fit(self):
        len_state = self.samples[0][0].size
        initial_parameters = np.zeros(len_state + 1)
        initial_parameters[-1] = 1
        constraints = ()
        for i in range(0, initial_parameters.size):
            if i == initial_parameters.size-1:
                constraints += ((0, None),)
            else:
                constraints += ((None, None),)
        res = minimize(self._calc_kl_distance,
                       initial_parameters,
                       jac=False,
                       method='SLSQP',
                       options={'ftol': 1e-6, 'disp': True})
        print(res)
        self.mean = res.x[:np.size(res.x)-1]
        self.var = res.x[-1]

    def _calc_kl_distance(self, parameters):
        states = self.samples[0]
        number_of_samples = np.shape(states)[0]
        len_state = states[0].size
        mean = parameters[0:len_state]
        covariance = parameters[-1]
        print(covariance)
        kl_distance = 0.0
        for i in range(0, number_of_samples):
            kl_distance += (1 / number_of_samples) * self.weights[i]\
                            * mvn.logpdf(states[i], mean, covariance)
        return kl_distance

    def act(self, state):
        act = mvn.pdf(state, self.mean, self.var)
        return act
