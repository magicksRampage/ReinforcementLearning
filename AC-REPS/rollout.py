import numpy as np


class Rollout:

    def __init__(self, states, actions, next_states, rewards):
        self.states = states
        self.state_length = np.shape(states[0])[0]
        self.actions = actions
        self.action_length = np.shape(actions[0])[0]
        self.next_states = next_states
        self.rewards = rewards
        self.length = np.shape(states)[0]
