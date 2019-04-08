import numpy as np


class Rollout:

    def __init__(self, states, actions, next_states, rewards):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.length = np.shape(states)[0]
