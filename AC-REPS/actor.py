import numpy as np
import model
import time
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf
from scipy.special import erfinv


class Actor:
    """

    """

    def __init__(self, rollouts, q_critic, v_critic, old_actor, progress):
        """

        :param rollout: The observations of the last roll-out containing (states, actions, following_states, rewards)
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
        self.model = model.Model(model.POLYNOMIAL_LINEAR,
                                 np.shape(self.rollouts[0].states[0])[0])
        self.std_deviation = 1.
        self.progress = progress
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
        initial_values = np.append(self.model.parameters, self.std_deviation)

        constraints = ()
        for i in range(0, initial_values.size):
            if i == initial_values.size-1:
                constraints += ((0, np.exp(-self.progress)),)
            else:
                constraints += ((-0.5, 0.5),)

        res = minimize(self._wrap_inputs,
                       initial_values,
                       jac=False,
                       method='SLSQP',
                       bounds=constraints,
                       options={'ftol': 1e-6, 'disp': False}
                       )
        print(res)

    def _wrap_inputs(self, values):
        self.model.parameters = values[0:-1]
        self.std_deviation = values[-1]
        return self._calc_regulated_kl_distance()

    def _calc_regulated_kl_distance(self):
        #---Profiling---
        start_time = time.clock()
        #---
        regulated_kl_distance = 0.0
        for i in range(0, np.size(self.rollouts)):
            states = self.rollouts[0].states
            actions = self.rollouts[0].actions
            number_of_samples = np.shape(states)[0]
            regulated_sum = 0.0
            # ---Profiling---
            start_rollout_time = time.clock()
            pi_time = 0.0
            # ---
            for j in range(0, number_of_samples):
                # ---Profiling---
                prev_time= time.clock()
                # ---
                pi = self._policy_probability(states[j], actions[j])
                # ---Profiling---
                pi_time += time.clock() - prev_time
                # ---
                if pi <= 0.0:
                    return np.inf
                regulated_sum += self.regulated_weights[j] * np.log(pi)
            # ---Profiling---
            # print("Relative time to calculate pi: ", np.round(pi_time / (time.clock() - start_rollout_time), 2))
            # ---
            # Ignore the regulator term as it is constant
            regulated_kl_distance += -regulated_sum * (1 / number_of_samples)  # +np.exp(weight_regulator)

        # print("Time used to calculate the regulated_kl_distance: ", time.clock()-start_time)
        return regulated_kl_distance

    def _policy_probability(self, state, action):
        mean = self.model.evaluate(state)
        return self._pseudo_gaussian_pdf(mean, self.std_deviation, action)

    def _pseudo_gaussian_pdf(self, mean, std_deviation, value):
        # print(mean, "|", std_deviation, "|", value)
        if std_deviation == 0.0:
            return float(np.abs(mean-value) < 1e-6)
        pdf = (np.exp(-(value - mean)**2 / (2 * std_deviation ** 2)) / np.sqrt(2 * np.pi * std_deviation ** 2))
        cdf_over_range = self._gaussian_cdf(mean, std_deviation, 1) - self._gaussian_cdf(mean, std_deviation, -1)
        if cdf_over_range < 1e-10:
            return 1.0
        result = pdf / cdf_over_range
        # print(result, "|", pdf, "|", cdf_inside_range)
        # print(cdf_inside_range)
        return result

    def _pseudo_gaussian_quantile(self, mean, std_deviation, x):
        lower_cdf = self._gaussian_cdf(mean, std_deviation, -1)
        upper_cdf = self._gaussian_cdf(mean, std_deviation, 1)
        regulated_sample = x*(upper_cdf - lower_cdf) + lower_cdf
        return self._gaussian_quantile(mean, std_deviation, regulated_sample)

    def _gaussian_cdf(self, mean, std_deviation, value):
        return (1/2) * (1 + erf((value - mean) / (2 * std_deviation**2)))

    def _gaussian_quantile(self, mean, std_deviation, x):
        return norm(loc=mean, scale=std_deviation).ppf(x)

    def act(self, state):
        mean = self.model.evaluate(state)
        sample = self._pseudo_gaussian_quantile(mean, self.std_deviation, np.random.random())
        redraws = 0
        while (sample < -1) | (1 < sample):
            redraws += 1
            # print(sample)
            sample = self._pseudo_gaussian_quantile(mean, self.std_deviation, np.random.random())
        # print("Drew ", redraws, " new samples")
        return np.clip(sample, -1, 1)
