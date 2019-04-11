import numpy as np


class Rollout:
    """
    Collection of the observation that define an episode.
    Episodes are traversed in steps which define a tuple (state, action, next_state, reward)
    The i-th step corresponds to the tuple (states[i], actions[i], next_states[i], rewards[i])

    Attributes:
        states (n x s): the states traverse in the episode
        state_length (int): dimension of a state
        actions (n x a): actions applied in the episode
        action_length (int): dimension of an action
        next_states (n x s): the states reach after applying an action in the episode
                             next_states[i] == states [i+1]
        rewards (n x 1): the direct rewards collected in an episode
        length: the length of the episode recorded in the object
    """

    def __init__(self, states, actions, next_states, rewards):
        """
        :param states (n x s): the states traverse in the episode
        :param actions (n x a): actions applied in the episode
        :param next_states (n x s): the states reach after applying an action in the episode
                                    next_states[i] == states [i+1]
        :param rewards (n x 1): rewards: the direct rewards collected in an episode
        """
        self.states = states
        self.state_length = np.shape(states[0])[0]
        self.actions = actions
        self.action_length = np.shape(actions[0])[0]
        self.next_states = next_states
        self.rewards = rewards
        self.length = np.shape(states)[0]
