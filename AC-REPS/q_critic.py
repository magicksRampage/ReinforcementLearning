import numpy as np
from scipy import optimize as opt
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim

# TODO: Hyper-parameter: Decay-rate for the rewards and Qs
DECAY = 0.98
# TODO: Computability: max_epoches
MAX_EPOCHS = 2500


class QCritic:

    def __init__(self, rollouts):
        self.rollouts = rollouts
        # Size of a state-action pair
        self.n_in = np.size(rollouts[0].states[0]) + np.size(rollouts[0].actions[0])
        self.n_h = 20
        # Size of an action
        self.n_out = np.size(rollouts[0].actions[0])
        # Feed the NN all the samples
        self.batch_size = rollouts[0].length
        # Init Neural Network
        self.model = nn.Sequential(nn.Linear(self.n_in, self.n_h),
                                   nn.ReLU(),
                                   nn.Linear(self.n_h, self.n_h),
                                   nn.ReLU(),
                                   nn.Linear(self.n_h, self.n_h),
                                   nn.ReLU(),
                                   nn.Linear(self.n_h, self.n_out))
        self.criterion = nn.MSELoss()
        # TODO: Hyper-parameter: learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self._fit()

    def _fit(self):
        states = self.rollouts[0].states
        actions = self.rollouts[0].actions
        next_states = self.rollouts[0].next_states
        rewards = torch.from_numpy(self.rollouts[0].rewards)

        model_input = np.zeros((self.batch_size,
                                np.size(states[0]) + np.size(actions[0]),))
        model_input[:, 0:np.shape(states)[1]] = states
        model_input[:, np.shape(states)[1]:] = actions
        model_input = torch.from_numpy(model_input)
        # Switch to float because the NN demands it ?.?
        model_input = model_input.float()
        rewards = rewards.float()
        loss = np.inf
        epoch = 0
        # TODO: Hyper-parameter: loss_threshold
        loss_threshold = 1e-6
        while (loss > loss_threshold) & (epoch < MAX_EPOCHS):
            # Forward Propagation
            predictions = self.model(model_input)
            number_of_predictions = predictions.size()[0]
            average_prediction = 0.0
            average_reward = 0.0
            for i in range(0, number_of_predictions):
                average_prediction += (1/number_of_predictions) * predictions[i]
                average_reward += (1/number_of_predictions) * rewards[i]

            # Define the Td-Q-Targets arcording to n-look-ahead in the sampels
            target = rewards.clone()
            td_range = 1
            for td_step in range(1, 1+td_range):
                if td_step == td_range:
                    target[0:target.size()[0] - td_step] += np.power(DECAY, td_step) * predictions[td_step:]
                else:
                    target[0:target.size()[0] - td_step] += np.power(DECAY, td_step) * rewards[td_step:]
            # Compute and print loss
            loss = self.criterion(predictions, target)
            epoch += 1
            # print('epoch: ', epoch, ' loss: ', loss.item(), ' Average Prediction: ', average_prediction)

            # Zero the gradients
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            if (loss < loss_threshold) | (epoch == MAX_EPOCHS):
                # print(rewards)
                print('epoch: ', epoch, ' loss: ', loss.item(), ' Average Prediction: ', average_prediction)
                loss.backward()
            else:
                loss.backward(retain_graph=True)

            # Update the parameters
            self.optimizer.step()
        # print([predictions, rewards])
        # print(number_of_predictions)

    def estimate_q(self, state, action):
        input = torch.from_numpy(np.reshape(np.append(state, action), (1, -1)))
        input = input.float()
        result = self.model(input)
        # Cast result to a number
        return result.detach().numpy()[0][0]


