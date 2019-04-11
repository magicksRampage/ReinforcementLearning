import numpy as np
import model
import time
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf
from scipy.special import erfinv


class Actor:
    """
    Defines the policy of our agent by fitting a model to the specifications of the enclosed
     AC-REPS related paper

     Attributes:
        rollouts (n x object): tuple containing "rollout.Rollout"-objects whom specify the rollouts
        q_critic (object): "q_critic.Q_critic"-object defining the approximation for the Q-function
        v_critiv (object): "v_critic.V_critic"-object defining the approximation for the V-function
        old_actor (object): "actor.Actor"-Object defining the previous policy
        regulated_weight_batches (n x rollout.length): the importance weights of the samples from the rollouts
        model (object): "model.Model"-Object defining how the observations translate into an action
        std_deviation (double): The standard deviation of the gaussian placed over the preferred action to ensure exploration
        progress (double): The part of the training process that is already completed (used to scale exploration)
    """

    def __init__(self, rollouts, q_critic, v_critic, old_actor, progress):
        """
        :param rollout (object): The observations of the last roll-out containing (states, actions, following_states, rewards)
        :param q_critic (object): The object estimating the Q-Function
        :param v_critic (object: The object estimating the V-Function
        :param old_actor (object): The object defining the previous policy

        Calls:
            _calculate_weights
            _fit
        """
        self.rollouts = rollouts
        self.q_critic = q_critic
        self.v_critic = v_critic
        # The old actor will be none in the first run so we explore randomly
        self.old_actor = old_actor
        self.regulated_weight_batches = ()
        self._calculate_weights()
        self.model = model.Model(model.POLYNOMIAL_LINEAR,
                                 np.shape(self.rollouts[0].states[0])[0])
        self.std_deviation = 1.0
        self.progress = progress
        self._fit()

    def _calculate_weights(self):
        """
        Calculates the importance weights of each sample

        Updates:
            regulated_weight_batches

        Calls:
            q_critic.estimate_v
            v_critiv.estimate_q

        :return: None
        """

        for i in range(0, np.size(self.rollouts)):
            states = self.rollouts[i].states
            actions = self.rollouts[i].actions
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
            self.regulated_weight_batches += (weights,)

    def _fit(self):
        """
        Fits the actor model to the specifications of the aforementioned paper

        Updates:
            model.parameters (through _wrap_inputs)
            std_deviation (through _wrap_inputs)

        Calls:
            scipy.optimize.minimize
            _wrap_inputs

        :return: None
        """
        print("Handing of work to scipy")
        prev_time = time.clock()
        initial_values = np.append(self.model.parameters, self.std_deviation)

        constraints = ()
        for i in range(0, initial_values.size):
            if i == initial_values.size-1:
                constraints += ((0, 1*np.exp(-self.progress)),)
            else:
                constraints += ((-1, 1),)

        res = minimize(self._wrap_inputs,
                       initial_values,
                       jac=False,
                       method='SLSQP',
                       bounds=constraints,
                       options={'ftol': 1e-6, 'disp': False}
                       )
        print(res)

    def _wrap_inputs(self, values):
        """
        Wraps inputs from the form scipy.optimize.minimize uses to the form of the programm

        :param values(n+1 x 1): the vector of values scipy optimizes (last entry contains std_deviation)

        Update:
            model.parameters
            std_deviation

        :return: the regulated kl_distance between the old policy and the new policy
        """
        self.model.parameters = values[0:-1]
        self.std_deviation = values[-1]
        return self._calc_regulated_kl_distance()

    def _calc_regulated_kl_distance(self):
        """
        Calculates the regulated kl_distance between the old policy and the new policy.
        It is not necessary to remove the regulation because both distance share the same minima (log-properties)

        Calls:
            _policy_probability

        :return: the regulated kl_distance
        """
        #---Profiling---
        start_time = time.clock()
        #---
        regulated_kl_distance = 0.0
        for i in range(0, np.size(self.rollouts)):
            rollout = self.rollouts[i]
            states = rollout.states
            actions = rollout.actions
            weights = self.regulated_weight_batches[i]
            number_of_samples = rollout.length
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
                regulated_sum += weights[j] * np.log(pi)
            # ---Profiling---
            # print("Relative time to calculate pi: ", np.round(pi_time / (time.clock() - start_rollout_time), 2))
            # ---
            # Ignore the regulator term as it is constant
            regulated_kl_distance += -regulated_sum * (1 / number_of_samples)  # +np.exp(weight_regulator)

        # print("Time used to calculate the regulated_kl_distance: ", time.clock()-start_time)
        return regulated_kl_distance

    def _policy_probability(self, state, action):
        """
        Calculates the probability that the policy would have produced a certain action at a certain state

        :param state (n x 1): the state at which the policy is evaluated
        :param action (m x 1): the action for which the probability is calculated

        Calls:
            _pseudo_gaussian_pdf

        :return: the probability of the action for a state under the current policy
        """
        mean = self.model.evaluate(state)
        return self._pseudo_gaussian_pdf(mean, self.std_deviation, action)

    def _pseudo_gaussian_pdf(self, mean, std_deviation, value):
        """
        Corrects the gaussian-pdf to completely reside in the interval [-1,1]

        :param mean (double): the mean of the original gaussian
        :param std_deviation (double): the std_deviation of the original gaussian
        :param value (double): the value for the pdf of the gaussian is to be evaluated

        Calls:
            _gaussian_cdf (this is faster than using a scipy.stats)

        :return: the modified probability-density-function
        """
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

    def _pseudo_gaussian_quantile(self, mean, std_deviation, value):
        """
        Corrects the inverse_gaussian_cdf to only produce outputs in the interval [-1,1]
        This is needed to produce outputs in the pseudo_gaussian

        :param mean (double): the mean of the original gaussian function
        :param std_deviation (double): the standard deviation of the original gaussian
        :param value (double): the probability value for which a sample of the pseudo_gaussian is to be evaluated

        :return:  a sample of the pseudo_gaussian
        """
        lower_cdf = self._gaussian_cdf(mean, std_deviation, -1)
        upper_cdf = self._gaussian_cdf(mean, std_deviation, 1)
        regulated_sample = value * (upper_cdf - lower_cdf) + lower_cdf
        return self._gaussian_quantile(mean, std_deviation, regulated_sample)

    def _gaussian_cdf(self, mean, std_deviation, value):
        """
        Calculates the gaussian cumulative density function.
        This is faster than using scipy.stats because their function "waste" computation on input sanitation

        :param mean (double): mean of the gaussian
        :param std_deviation (double): standard deviation of the gaussian
        :param value (double): value for which the cumulative densitiy function is to be evaluated

        Calls:
            scipy.special.erf

        :return: the density_function at the value for a defined gaussian distribution
        """
        return (1/2) * (1 + erf((value - mean) / (2 * std_deviation**2)))

    def _gaussian_quantile(self, mean, std_deviation, value):
        """
        Calculates the gaussian inverse cumulative density function (quantile).
        It would be preferable the avoid using scipy.state.norm bacause its slow,
        but the scipy.special.inverf is not numerically stable enough for our purposes.

        :param mean (double): mean of the gaussian
        :param std_deviation (double): standard deviation of the gaussian
        :param value (double): value for which the cumulative densitiy function is to be evaluated

        calls: scipy.special.inverf

        :return: the quantile at the value for a defined gaussian distribution
        """
        return norm(loc=mean, scale=std_deviation).ppf(value)

    def act(self, state):
        """
        Calculates an action for a given state based on a given modell.
        This action is then taken as the mean of a pseudo-gaussian over the action space.
        Thus the policy is not deterministic, but its degree of exploration is well defined through
        self.std_deviation

        :param state (n x 1): the state for which an action is to be taken

        :return: an normalized action on the interval [-1,1]
        """
        mean = self.model.evaluate(state)
        sample = self._pseudo_gaussian_quantile(mean, self.std_deviation, np.random.random())
        redraws = 0
        while (sample < -1) | (1 < sample):
            redraws += 1
            # print(sample)
            sample = self._pseudo_gaussian_quantile(mean, self.std_deviation, np.random.random())
        # print("Drew ", redraws, " new samples")
        return np.clip(sample, -1, 1)
