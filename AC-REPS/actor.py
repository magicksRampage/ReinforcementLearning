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
from scipy.stats import norm


class Actor:

    def __init__(self, rollouts, q_critic, v_critic, old_actor):
        """

        :param samples: The observations of the last roll-out containing (states, actions, following_states, rewards)
        :param q_critic: The object estimating the Q-Function
        :param v_critic: The object estimating the V-Function
        :param old_actor: The object defining the previous policy
        """
        self.rollouts = rollouts
        self.q_critic = q_critic
        self.v_critic = v_critic
        # The old actor will be none in the first run so we explore randomly
        self.old_actor = old_actor
        self.regulated_weights = None
        self._calculate_weights()
        self.model = model.Model(model.RANDOM_RBFS,
                                 np.shape(self.rollouts[0].states[0])[0],
                                 number_of_basis_functions=5)
        self.variance = 1.0
        self._fit()

    def _calculate_weights(self):
        states = self.rollouts[0].states
        actions = self.rollouts[0].actions
        number_of_samples = np.shape(states)[0]
        vc = self.v_critic
        qc = self.q_critic
        weights = np.zeros((number_of_samples, 1))
        weight_regulator = -np.inf
        # Numeric Regulation
        for i in range(0, number_of_samples):
            exponent = (qc.estimate_q(states[i], actions[i]) - vc.estimate_v(states[i])) / vc.eta
            if exponent > weight_regulator:
                weight_regulator = exponent
        for i in range(0, number_of_samples):
            # TODO: Ensure stability
            exponent = (qc.estimate_q(states[i], actions[i]) - vc.estimate_v(states[i])) / vc.eta
            weights[i] = np.exp(exponent - weight_regulator)
        self.regulated_weights = weights

    def _fit(self):
        print("Handing of work to scipy")
        prev_time = time.clock()
        initial_values = np.append(self.model.parameters, [self.variance])
        constraints = ()
        for i in range(0, initial_values.size):
            constraints += ((0, None),)
        res = minimize(self._wrap_inputs,
                       initial_values,
                       jac=False,
                       method='SLSQP',
                       bounds=constraints,
                       options={'ftol': 1e-6, 'disp': True})
        self.model.parameters = res.x[0:-1]
        print(res)
        print("Fitting Actor_Time: ", time.clock() - prev_time)

    def _wrap_inputs(self, values):
        self.model.parameters = values[0:-1]
        self.variance = values[-1]
        return self._calc_regulated_kl_distance()

    def _calc_regulated_kl_distance(self):
        states = self.rollouts[0].states
        actions = self.rollouts[0].actions
        number_of_samples = np.shape(states)[0]
        regulated_sum = 0.0
        for i in range(0, number_of_samples):
            pi = self._policy_probability(states[i], actions[i])
            if pi <= 0.0:
                return np.inf
            regulated_sum += self.regulated_weights[i] * np.log(self._policy_probability(states[i], actions[i]))
        # Ignore the regulator term as it is constant
        regulated_kl_distance = -regulated_sum * (1 / number_of_samples)  # +np.exp(weight_regulator)
        return regulated_kl_distance

    def _policy_probability(self, state, action):
        mean = self.model.evaluate(state)
        if self.variance <= 0.0:
            return np.abs(mean - action) < 1e-6
        return norm(loc=mean, scale=self.variance).pdf(action)

    def act(self, state):
        mean = self.model.evaluate(state)
        return np.clip(norm.rvs(loc=mean, scale=self.variance), 0, 1)
