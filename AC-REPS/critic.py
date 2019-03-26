import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
                                   nn.Linear(self.n_h, self.n_h),
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
        rewards = self.samples[3]
        model_input = np.zeros((self.batch_size,
                                np.size(states[0]) + np.size(actions[0]),))
        model_input[:, 0:np.shape(states)[1]] = states
        model_input[:, np.shape(states)[1]:] = actions
        # TODO: Fill in rest off copied code
        for epoch in range(50):
            # Forward Propagation
            y_pred = self.model(model_input)

            target = None
            # Compute and print loss
            loss = self.criterion(y_pred, target)
            print('epoch: ', epoch, ' loss: ', loss.item())

            # Zero the gradients
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            self. optimizer.step()


class VCritic:

    def __init__(self, value_parameters):
        self.parameters = value_parameters
