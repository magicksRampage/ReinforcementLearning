import gym
import time
import numpy as np
import actor as ac
import q_critic
import v_critic
import quanser_robots


INITIAL_ETA = 1
EPISODE_LENGTH = 100


class Agent:

    def __init__(self, env):
        self.environment_name = env
        env = gym.make(self.environment_name)
        self.state_dimensions = np.reshape(np.append(env.observation_space.low, env.observation_space.high), (2, -1)).transpose()
        self.action_dimension = [env.action_space.low[0], env.action_space.high[0]]
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
        self.q_critic = q_critic.QCritic(self.samples)
        print("Q-Critic fitted")
        self.v_critic = v_critic.VCritic(self.samples, self.q_critic)
        print("V-Critic minimized dual")
        self.actor = ac.Actor(self.samples, self.q_critic, self.v_critic, self.actor)

    def generate_episode(self):
        env = gym.make(self.environment_name)
        norm_prev_obs = env.reset()
        # Normalize the observations
        for i in range(0, np.shape(self.state_dimensions)[0]):
            norm_prev_obs[i] = self._normalize(norm_prev_obs[i], self.state_dimensions[i][0], self.state_dimensions[i][1])

        norm_states = ()
        norm_actions = ()
        norm_next_states = ()
        rewards = ()
        if self.samples is None:
            samples = ()
        else:
            samples = self.samples

        for i in range(0, EPISODE_LENGTH):
            if self.actor is None:
                # If you haven't trained an actor explore randomly
                action = self._denormalize(np.random.random(), self.action_dimension[0], self.action_dimension[1])
            else:
                action = self._denormalize(self.actor.act(norm_prev_obs), self.action_dimension[0], self.action_dimension[1])
            norm_obs, reward, done, info = env.step(np.array([action]))
            # Normalize the observations
            for i in range(0, np.shape(self.state_dimensions)[0]):
                norm_obs[i] = self._normalize(norm_obs[i], self.state_dimensions[i][0], self.state_dimensions[i][1])
            norm_states += (norm_prev_obs,)
            norm_actions += (self._normalize(action, self.action_dimension[0], self.action_dimension[1]),)
            norm_next_states += (norm_obs,)
            rewards += (reward,)
            norm_prev_obs = norm_obs
            env.render()
        env.close()

        number_of_samples = np.shape(norm_states)[0]
        # Reshape arrays to catch shape of (n,) and replace it with (n,1)
        samples += (np.array(np.reshape(norm_states, (number_of_samples, -1))),
                    np.array(np.reshape(norm_actions, (number_of_samples, -1))),
                    np.array(np.reshape(norm_next_states, (number_of_samples, -1))),
                    np.array(np.reshape(rewards, (number_of_samples, -1))))
        self.samples = samples

    def _normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def _denormalize(self, normalized_value, min_value, max_value):
        return (normalized_value * (max_value - min_value)) + min_value



