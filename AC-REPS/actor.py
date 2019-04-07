"""
Simple class determining and defining a policy.

Functions
---------
- act : returns an action given a state
"""
import numpy as np
import model
import time
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
        self.model = model.Model(model.RANDOM_RBFS,
                                 np.shape(self.samples[0][0])[0])
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
        print("Handing of work to scipy")
        prev_time = time.clock()
        initial_values = self.model.parameters
        constraints = ()
        for i in range(0, initial_values.size):
            constraints += ((0, None),)
        res = minimize(self._calc_kl_distance,
                       initial_values,
                       jac=False,
                       method='SLSQP',
                       bounds=constraints,
                       options={'ftol': 1e-6, 'disp': True})
        self.model.parameters = res.x
        print(res)
        print("Fitting Actor_Time: ", time.clock() - prev_time)

    def _calc_kl_distance(self, parameters):
        self.model.parameters = parameters
        states = self.samples[0]
        number_of_samples = np.shape(states)[0]
        kl_distance = 0.0
        for i in range(0, number_of_samples):
            if self.model.evaluate(states[i]) <= 0.0:
                print("model unfit for optimizing over kl-distance")
            kl_distance += (1 / number_of_samples) * self.weights[i]\
                            * np.log(self.model.evaluate(states[i]))
        return kl_distance

    def act(self, state):
        act = self.model.evaluate(state)
        return act
