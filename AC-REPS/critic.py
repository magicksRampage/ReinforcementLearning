import numpy as np
from scipy import optimize as opt
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim

# TODO: Handle Hyperparameters correctly
DECAY = 0.9
EPSILON = 0.5
INITIAL_ETA = 1.0
INITIAL_PARAMETER_SCALE = 0.0


class QCritic:

    def __init__(self, samples):
        self.samples = samples
        # Size of a state-action pair
        self.n_in = np.size(samples[0][0]) + np.size(samples[1][0])
        self.n_h = 20
        # Size of an action
        self.n_out = np.size(samples[1][0])
        # Feed the NN all the samples
        self.batch_size = np.shape(samples[1])[0]
        # Init Neural Network
        self.model = nn.Sequential(nn.Linear(self.n_in, self.n_h),
                                   nn.ReLU(),
                                   nn.Linear(self.n_h, self.n_out),
                                   nn.Sigmoid())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self._fit()

    def _fit(self):
        states = self.samples[0]
        actions = self.samples[1]
        next_states = self.samples[2]

        rewards = torch.from_numpy(self.samples[3])
        model_input = np.zeros((self.batch_size,
                                np.size(states[0]) + np.size(actions[0]),))
        model_input[:, 0:np.shape(states)[1]] = states
        model_input[:, np.shape(states)[1]:] = actions
        model_input = torch.from_numpy(model_input)
        # Switch to float because the NN demands it ?.?
        model_input = model_input.float()
        rewards = rewards.float()
        # TODO: Is the loss correct here?
        # TODO: Specify Hyperparameter (learning-rate, epoches)
        number_of_epochs = 50
        for epoch in range(0, number_of_epochs):
            # Forward Propagation
            predictions = self.model(model_input)

            target = rewards.clone()
            target[0:target.size()[0] - 1] += DECAY * predictions[1:]
            # Compute and print loss
            loss = self.criterion(predictions, target)
            print('epoch: ', epoch, ' loss: ', loss.item())

            # Zero the gradients
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            if epoch == (number_of_epochs - 1):
                loss.backward()
            else:
                loss.backward(retain_graph=True)

            # Update the parameters
            self.optimizer.step()

        self.estimate_q(states[0], actions[0])

    def estimate_q(self, state, action):
        input = torch.from_numpy(np.reshape(np.append(state, action), (1, -1)))
        input = input.float()
        result = self.model(input)
        # Cast result to a number
        return result.detach().numpy()[0][0]


class VCritic:

    def __init__(self, samples, q_critic):
        self.samples = samples
        self.q_critic = q_critic
        self.eta = None
        self.parameters = None
        self._initialize_parameters()
        self._minimize_dual()

    def _initialize_parameters(self):
        self.eta = INITIAL_ETA
        # Polynomial n == 3
        len_state = np.shape(self.samples[0])[1]
        len_parameters = len_state + np.power(len_state, 2)
        # + np.power(len_state, 3)
        self.parameters = INITIAL_PARAMETER_SCALE * np.ones((len_parameters,))

    def _minimize_dual(self):
        initial_values = np.append(self.parameters, self.eta)
        constraints = ()
        for i in range(0, initial_values.size):
            if i == initial_values.size-1:
                constraints += ((0, None),)
            else:
                constraints += ((-100, 100),)
        # TODO: Find a scipy-configuration that stably minimizes the dual
        res = minimize(self._wrap_inputs,
                       initial_values,
                       method='trust-constr',
                       bounds=constraints,
                       jac='2-point',
                       hess=opt.BFGS,
                       options={'verbose': 2})

        self.eta = res.x[-1]
        self.parameters = res.x[0:res.x.size-1]
        print(res)

    def _wrap_inputs(self, arg):
        return self.evaluate_dual(arg[-1], arg[0:arg.size-1])

    def evaluate_dual(self, eta=None, parameters=None):
        if eta is None:
            eta = self.eta
        if eta == 0:
            return np.inf
        if parameters is None:
            parameters = self.parameters
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
        exp_regulator = running_max_value - 3
        for i in range(0, number_of_samples):
            average_state += (1 / number_of_samples) * states[i]

            exp_sum += np.exp(((self.q_critic.estimate_q(states[i], actions[i]) - self.estimate_v(states[i])) / eta)
                              - exp_regulator)
        dual = 0.0
        dual += EPSILON*eta
        dual += self.estimate_v(average_state, parameters)
        dual += eta * (np.log(exp_sum) - np.log(number_of_samples) + exp_regulator)
        # print(parameters, eta)
        return dual

    def estimate_v(self, state, parameters=None):
        if parameters is None:
            parameters = self.parameters
        # Establish basis-functions-evaluations
        basis_evaluations = ()
        # In this case polonomiyal features of degree 3
        len_state = state.size
        # n == 1
        for i in range(0, len_state):
            basis_evaluations += (state[i],)
        # n == 2
        for i in range(0, len_state):
            for j in range(0, len_state):
                basis_evaluations += (state[i]*state[j],)
        """
        # n == 3
        for i in range(0, len_state):
            for j in range(0, len_state):
                for k in range(0, len_state):
                    basis_evaluations += (state[i]*state[j]*state[k],)
        """

        """
        if not parameters[0] == 0:
            print("Stop")
        """
        return np.dot(parameters, np.array(basis_evaluations))
