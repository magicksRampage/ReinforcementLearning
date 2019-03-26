import numpy as np


class Actor:

    def __init__(self, min_action, max_action, eta, q_critic, v_critic, samples):
        self.min_action = min_action
        self.max_action = max_action
        self.eta = eta
        self.q_distribution = q_critic
        self.v_distribution = v_critic
        self.samples = samples
