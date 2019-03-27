import numpy as np


class Actor:

    def __init__(self, min_action, max_action, samples, q_critic, v_critic):
        self.min_action = min_action
        self.max_action = max_action
        self.q_distribution = q_critic
        self.v_distribution = v_critic
        self.samples = samples

    def act(self):
        return None
