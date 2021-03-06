import gym
import time
import numpy as np
import actor as ac
import rollout as rl
import q_critic
import v_critic
import quanser_robots


MAX_EPISODE_LENGTH = 200
NUMBER_OF_BATCHES = 5
NUMBER_OF_EPISODES = 50

class Agent:
    """
    The upper-most abstraction layer of the reinforcement-learning process.
    Handles all transitional information between different steps for calculating a new policy
    and between policies

    Attributes:
        env (object): the object returned from gym.make()
        state_dimensions (n x 2): matrix containing the bounds on the observation dimensions
        action_dimensions (m x 2): matrix containing the bounds on the action dimensions
        rollouts (l x object): tuple containing "rollout.Rollout"-objects whom specify the rollouts
        q_critic (object): "q_critic.Q_critic"-object defining the approximation for the Q-function
        v_critiv (object): "v_critic.V_critic"-object defining the approximation for the V-function
        actor (object): "actor.Actor"-object defining the gaussian policy
    """

    def __init__(self, environment_name):
        """
        :param environment_name (String): The String defining the gym.environment the agent will act in

        Calls:
            gym.make
        """
        self.env = gym.make(environment_name)
        self.state_dimensions = np.reshape(np.append(self.env.observation_space.low, self.env.observation_space.high), (2, -1)).transpose()
        self.action_dimension = [self.env.action_space.low[0], self.env.action_space.high[0]]
        # Will be containing the values of the most recent train_step()
        self.rollouts = ()
        self.q_critic = None
        self.v_critic = None
        self.actor = None
        self.average_rewards = ()

    def train(self):
        """
        Train a policy over a fixed number of iterations:

        Calls:
            train_step

        :return: None
        """
        converged = False
        for i in range(0, NUMBER_OF_EPISODES):
            # TODO: define conversion target
            print("")
            print("---------------------")
            print("---------------------")
            print("---- Episode ", i+1, "----")
            print("---------------------")
            print("---------------------")
            print("")
            prev_time = time.clock()
            self.train_step(i / NUMBER_OF_EPISODES)
            print("")
            print("---------------------")
            print("----Time Elapsed for the episode: ", int(time.clock()-prev_time), "----")
            print("---------------------")
        print(self.average_rewards)

    def train_step(self, progress):
        """
        Update policy through the following 4 steps:
            1. Generate a batch of episodes
            2. Define Q through Critic
            3. Define V by minimizing the dual
            4. Define a new policy by fitting an actor

        Updates:
            self.q_critic
            self.v_critic
            self.actor

        Calls:
            generate_episode
            q_critic.Q_Critic
            v_critic.V_Critic
            actor.Actor

        :return: None
        """

        prev_time = time.clock()
        for i in range(0, NUMBER_OF_BATCHES):
            self.generate_episode()
        print("---Generated Episodes in ", int(time.clock()-prev_time), " Seconds---")

        n = 0.0
        reward_sum = 0.0
        for i in range(0, NUMBER_OF_BATCHES):
            rollout = self.rollouts[0]
            for j in range(0, rollout.length):
                reward_sum += rollout.rewards[j]
                n += 1
        average_reward = reward_sum / n

        print("---------------------")
        print("----Average Reward: ", average_reward, "----")
        print("---------------------")

        prev_time = time.clock()
        self.q_critic = q_critic.QCritic(self.rollouts)
        print("---------------------")
        print("---Q-Critic fitted in ", int(time.clock()-prev_time), " Seconds---")
        print("---------------------")

        prev_time = time.clock()
        self.v_critic = v_critic.VCritic(self.rollouts, self.q_critic)
        print("---------------------")
        print("---V-Critic fitted in ", int(time.clock()-prev_time), " Seconds---")
        print("---------------------")

        prev_time = time.clock()
        self.actor = ac.Actor(self.rollouts, self.q_critic, self.v_critic, self.actor, progress)
        print("---------------------")
        print("---Policy fitted in ", int(time.clock()-prev_time), " Seconds---")
        print("---------------------")

        self.rollouts = ()
        self.average_rewards += (average_reward,)

    def generate_episode(self):
        """
        Generate and save a single rollout with the current policy

        Updates:
            self.rollouts

        Calls:
            _normalize
            _denormalize
            gym.environment.reset
            gym.environment.step
            gym.environment.render
            gym.environment.close

        :return: None
        """
        norm_prev_obs = self.env.reset()
        # Normalize the observations
        for i in range(0, np.shape(self.state_dimensions)[0]):
            norm_prev_obs[i] = self._normalize(norm_prev_obs[i], self.state_dimensions[i][0], self.state_dimensions[i][1])

        norm_states = ()
        norm_actions = ()
        norm_next_states = ()
        rewards = ()

        done = False
        steps = 0
        while (not done) & (steps < MAX_EPISODE_LENGTH):
            if self.actor is None:
                # If you haven't trained an actor explore with a set gaussian == N(0,1)
                action = self._denormalize(np.clip(np.random.normal(0, 1), -1, 1),
                                           self.action_dimension[0],
                                           self.action_dimension[1])
            else:
                action = self._denormalize(self.actor.act(norm_prev_obs),
                                           self.action_dimension[0],
                                           self.action_dimension[1])
            norm_obs, reward, done, info = self.env.step(np.array([action]))
            # print(done)
            # Normalize the observations
            for i in range(0, np.shape(self.state_dimensions)[0]):
                norm_obs[i] = self._normalize(norm_obs[i],
                                              self.state_dimensions[i][0],
                                              self.state_dimensions[i][1])
            norm_states += (norm_prev_obs,)
            norm_actions += (self._normalize(action,
                                             self.action_dimension[0],
                                             self.action_dimension[1]),)
            norm_next_states += (norm_obs,)
            rewards += (reward,)
            norm_prev_obs = norm_obs
            self.env.render()
            steps += 1
        self.env.close()

        number_of_samples = np.shape(norm_states)[0]
        # Reshape arrays to catch shape of (n,) and replace it with (n,1)
        rollout = rl.Rollout(np.array(np.reshape(norm_states, (number_of_samples, -1))),
                             np.array(np.reshape(norm_actions, (number_of_samples, -1))),
                             np.array(np.reshape(norm_next_states, (number_of_samples, -1))),
                             np.array(np.reshape(rewards, (number_of_samples, -1))))
        self.rollouts += (rollout,)

    def _normalize(self, value, min_value, max_value):
        """
        Scales a value into the interval [-1,1]

        :param: value (double): The value to be scaled
        :param: min_value (double): The lower bound of the interval the value originated from
        :param: max_value (double): The upper bound of the interval the value originated from

        :return: The normalized Value
        """
        return ((value - min_value) / (max_value - min_value)) * 2 - 1

    def _denormalize(self, normalized_value, min_value, max_value):
        """
        Scales a up from the interval [-1,1]

        :param: value (double): The value to be scaled
        :param: min_value (double): The lower bound of the interval the value will be scaled to
        :param: max_value (double): bound of the interval the value will be scaled to

        :return: The denormalized Value
        """
        return (((normalized_value + 1) / 2) * (max_value - min_value)) + min_value



