import numpy as np
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
        self.fit()

    def fit(self):
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
        number_of_epochs = 500
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
        self.eta = INITIAL_ETA
        # Linear Features require len(state) parameters
        self.parameters = INITIAL_PARAMETER_SCALE * np.ones((np.shape(samples[0])[1],))
        self.minimize_dual()

    def minimize_dual(self):
        initial_values = np.append(self.parameters, self.eta)
        constraints = ()
        for i in range(0, initial_values.size):
            if i == initial_values.size-1:
                constraints += ((0, None),)
            else:
                constraints += ((None, None),)
        res = minimize(self._wrap_inputs,
                       initial_values,
                       method='SLSQP',
                       bounds=constraints,
                       options={'ftol': 1e-6, 'disp': True})

        self.eta = res.x[-1]
        self.parameters = res.x[0:res.x.size-1]

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
        running_average = 0.0
        exp_regulator = 0.0
        running_max_value = -np.inf
        for i in range(0, number_of_samples):
            exponent = ((self.q_critic.estimate_q(states[i], actions[i])- self.estimate_v(states[i])) / eta)
            if exponent > running_max_value:
                running_max_value = exponent
        exp_regulator = running_max_value - 3
        for i in range(0, number_of_samples):
            running_average += (1 / number_of_samples) * self.estimate_v(states[i], parameters)

            exp_sum += np.exp(((self.q_critic.estimate_q(states[i], actions[i])- self.estimate_v(states[i]))/ eta)
                       - exp_regulator)
        dual = EPSILON*eta + running_average + eta * (np.log(exp_sum) - np.log(number_of_samples) + exp_regulator)
        return dual

    def estimate_v(self, state, parameters=None):
        if parameters is None:
            parameters = self.parameters
        # Linear Features for the moment
        return np.dot(parameters, state)
