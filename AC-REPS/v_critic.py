import numpy as np
import model
import time
from scipy import optimize as opt
from scipy.optimize import minimize

# TODO: Hyper-parameters
EPSILON = 0.5
INITIAL_ETA = 1.0
INITIAL_PARAMETER_SCALE = 0.0


class VCritic:
    """
    The abstraction of the V-state-value-function
    This function is approximated by a linear model of choice

    Attributes:
        rollouts (object): "rollout.Rollout"-Object containing the episodes of samples
        q_critic (object): "q_critic.Q_critic"-object defining the approximation for the Q-function
        eta (float): scalar lagrangian multiplier
        model (object): "model.Model"-Object defining how the observations translate into an action

    """

    def __init__(self, rollouts, q_critic):
        """
        :param rollout (object): The observations of the last roll-out containing (states, actions, following_states, rewards)
        :param q_critic (object): The object estimating the Q-Function

        Calls:
            model.Model
            _minimize_dual
        """
        self.rollouts = rollouts
        self.q_critic = q_critic
        self.eta = INITIAL_ETA
        self.model = model.Model(model.POLYNOMIAL_LINEAR,
                                 np.shape(self.rollouts[0].states[0])[0])
        self._minimize_dual()

    def _minimize_dual(self):
        """
        Minimize the lagrangian dual through the model parameters

        Updates:
            model.parameters (through _wrap_inputs)
            eta (through _wrap_inputs)

        Calls:
            scipy.optimize.minimize
            _wrap_inputs
            estimate_v

        :return: None
        """
        print("Minimizing Dual")
        initial_values = np.append(self.model.parameters, self.eta)
        constraints = ()
        for i in range(0, initial_values.size):
            if i == initial_values.size-1:
                constraints += ((0, None),)
            else:
                constraints += ((None, None),)
        # TODO: Find a scipy-configuration that stably minimizes the dual
        print("Handing of work to scipy")
        prev_time = time.clock()
        res = minimize(self._wrap_inputs,
                       initial_values,
                       method='SLSQP',
                       bounds=constraints,
                       options={'disp': False})
        print(res)
        number_of_samples = self.rollouts[0].length
        average_v = 0.0
        for i in range(0, number_of_samples):
            average_v = (1/number_of_samples) * self.estimate_v(self.rollouts[0].states[i])

        print("Average V: ", average_v)

    def _wrap_inputs(self, values):
        """
        Wraps inputs from the form scipy.optimize.minimize uses to the form of the programm


        :param values(n+1 x 1): the vector of values scipy optimizes
                                                  (last entry contains eta)

        :return: the evaluation of the dual
        """
        self.model.parameters = values[0:-1]
        self.eta = values[-1]
        return self.evaluate_dual(values[-1], values[0:values.size - 1])

    def evaluate_dual(self):
        """
        Evaluate the lagrangian dual.
        Due to numerical issues an regulator for the exp_sum is used

        Calls:
            q_critic.estimate_q
            v_critic.estimate_v

        :return: the dual value
        """
        prev_time = time.clock()
        eta = self.eta
        if eta == 0:
            return np.inf

        # ---Profiling---
        start_time = time.clock()
        regulator_time = 0.0
        estimating_time = 0.0
        # ---
        dual = 0.0
        for i in range(0, np.size(self.rollouts)):
            number_of_samples = self.rollouts[i].length
            states = self.rollouts[i].states
            actions = self.rollouts[i].actions
            exp_sum = 0.0
            average_state = np.zeros(np.shape(states[0]))
            running_max_value = -np.inf
            exponents = np.zeros((number_of_samples, 1))
            for j in range(0, number_of_samples):
                exponents[j] = ((self.q_critic.estimate_q(states[j], actions[j]) - self.estimate_v(states[j])) / eta)
                average_state += (1 / number_of_samples) * states[j]
            exp_regulator = np.max(exponents)
            regulated_exponents = np.exp(exponents - exp_regulator)

            dual += EPSILON * eta
            dual += self.estimate_v(average_state)
            dual += eta * (np.log(np.sum(regulated_exponents)) - np.log(number_of_samples) + exp_regulator)

        # print(".")
        # print(parameters, eta)
        # print("Time used to calculate the dual: ", time.clock()-start_time)
        return dual

    def estimate_v(self, state):
        """
        Evaluate the model for a state

        :param state (n x 1): the state for which v is to be estimated

        Calls:
            model.evaluate

        :return: Estimation of the value function
        """
        res = self.model.evaluate(state)
        return res

