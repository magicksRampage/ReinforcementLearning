import gym
import time
import numpy as np
import actor as ac
import critic as cr
import quanser_robots


INITIAL_ETA = 1
EPISODE_LENGTH = 100
MEMORY_SIZE = 500


class Agent:

    def __init__(self, env):
        self.environment_name = env
        env = gym.make(self.environment_name)
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        env.close()
        # Will be containing the values of the most recent train_step()
        self.samples = None
        self.q_critic = None
        self.v_critic = None
        self.actor = None

    def train(self):
        converged = False
        for i in range(0, 50):
            # TODO: define conversion target
            self.train_step()

    def train_step(self):
        """
        Update policy through the following 4 steps:
            1. Generate an episode
            2. Define Q through Critic
            3. Define V by minimizing the dual
            4. Define a new policy through actor
        :return:
        """

        self.generate_episode()
        self.q_critic = cr.QCritic(self.samples)
        self.v_critic = cr.VCritic(self.samples, self.q_critic)
        self.actor = ac.Actor(self.min_action, self.max_action, self.samples, self.q_critic, self.v_critic)

    def generate_episode(self):
        env = gym.make(self.environment_name)
        prev_obs = env.reset()
        states = ()
        actions = ()
        next_states = ()
        rewards = ()
        if self.samples is None:
            samples = ()
        else:
            samples = self.samples

        for i in range(0, EPISODE_LENGTH):
            if self.samples is None:
                # Assume that no actor exists if no samples are gathered yet
                action = 2.0 * (self.min_action * np.random.random()) + self.max_action
            else:
                # Assume that an actor exists otherwise
                action = self.actor.act()
            obs, reward, done, info = env.step(np.array(action))
            states += (prev_obs,)
            actions += (action,)
            next_states += (obs,)
            rewards += (reward,)
            prev_obs = obs
            env.render()

        number_of_samples = np.shape(states)[0]
        # Reshape arrays to catch shape of (n,) and replace it with (n,1)
        samples += (np.array(np.reshape(states, (number_of_samples, -1))),
                    np.array(np.reshape(actions, (number_of_samples, -1))),
                    np.array(np.reshape(next_states, (number_of_samples, -1))),
                    np.array(np.reshape(rewards, (number_of_samples, -1))))
        self.samples = samples


