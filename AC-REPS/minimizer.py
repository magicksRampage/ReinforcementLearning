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

    def __init__(self, min_action, max_action, samples, q_critic):
        self.min_action = min_action
        self.max_action = max_action
        self.samples = samples
        self.q_critic = q_critic
        self.eta = INITIAL_ETA
        self.model = model.Model(model.POLYNOMIAL_CUBIC,
                                 np.shape(self.samples[0][0])[0],
                                 min_in=self.min_action,
                                 max_in=self.max_action)
        """
        self.parameters = None
        self._initialize_parameters()
        """
        self._minimize_dual()

    def _minimize_dual(self):
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
                       options={'disp': True})
        self.eta = res.x[-1]
        self.parameters = res.x[0:res.x.size-1]
        print(res)
        print("Time: ", time.clock() - prev_time)

    def _wrap_inputs(self, arg):
        return self.evaluate_dual(arg[-1], arg[0:arg.size-1])

    def evaluate_dual(self, eta=None, parameters=None):
        if eta is None:
            eta = self.eta
        if eta == 0:
            return np.inf
        if parameters is not None:
            self.model.parameters = parameters
        number_of_samples = np.shape(self.samples[0])[0]
        states = self.samples[0]
        actions = self.samples[1]
        exp_sum = 0.0
        average_state = np.zeros(np.shape(states[0]))
        exp_regulator = 0.0
        running_max_value = -np.inf
        for i in range(0, number_of_samples):
            exponent = ((self.q_critic.estimate_q(states[i], actions[i]) - self.estimate_v(states[i])) / eta)
            if exponent > running_max_value:
                running_max_value = exponent
        exp_regulator = running_max_value + np.log(number_of_samples)
        for i in range(0, number_of_samples):
            average_state += (1 / number_of_samples) * states[i]

            exp_sum += np.exp(((self.q_critic.estimate_q(states[i], actions[i]) - self.estimate_v(states[i])) / eta)
                              - exp_regulator)
        dual = 0.0
        dual += EPSILON*eta
        dual += self.estimate_v(average_state)
        dual += eta * (np.log(exp_sum) - np.log(number_of_samples) + exp_regulator)
        # print(".")
        # print(parameters, eta)
        return dual

    def estimate_v(self, state):
        res = self.model.evaluate(state)
        return res

