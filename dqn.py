import numpy as np
import random as random
from collections import deque

from cnn_target import CNNtarget

# See https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf for model description


class DQN:
    def __init__(self, num_actions, observation_shape, dqn_params, cnn_params, prog_params):
        self.num_actions = num_actions
        self.epsilon = dqn_params['epsilon']
        self.gamma = dqn_params['gamma']
        self.mini_batch_size = dqn_params['mini_batch_size']
        self.observation_size = 1
        for a in observation_shape:
            self.observation_size = self.observation_size * a
        self.num_observations = cnn_params['num_observations']
        self.verbose = prog_params['verbose']
        self.tensorboard = prog_params['tensorboard']

        # memory
        self.memory = deque(maxlen=dqn_params['memory_capacity'])
        self.observations = np.zeros(self.num_observations*self.observation_size)

        # initialize network
        self.model = CNNtarget(self.num_actions, self.observation_size, cnn_params,
                               verbose = self.verbose, tensorboard = self.tensorboard)

    def select_action(self, observation):
        """
        Selects the next action to take based on the current state and learned Q.

        Args:
            observation: the current state

        """

        # First get the observation history
        obs_with_history = self.observations

        if random.random() < self.epsilon:
            # with epsilon probability select a random action
            action = np.random.randint(0, self.num_actions)
        else:
            # select the action a which maximizes the Q value
            obs = np.array([obs_with_history])
            q_values = self.model.predict(obs)
            action = np.argmax(q_values)

        return action

    def update_observation_history(self, new_observation):
        """
        Takes an observation as input. Updates the observation history with the latest observation.
        Returns the new observation with history.

        Args:
            observation: the raw current state

        """
        
        # First translate history from one "observation_size"
        self.observations[0:-1*self.observation_size] = self.observations[self.observation_size:]
        # Then store the last observation to the end
        self.observations[-1*self.observation_size:] = new_observation

        return self.observations

    def update_state(self, action, new_observation, reward, done):
        """
        Stores the most recent action in the replay memory.

        Args:
            action: the action taken
            new_observation: the state after the action is taken
            reward: the reward from the action
            done: a boolean for when the episode has terminated

        """
        obs_with_history = self.observations
        new_obs_with_history = self.update_observation_history(new_observation)

        transition = {'action': action,
                      'observation': obs_with_history,
                      'new_observation': new_obs_with_history,
                      'reward': reward,
                      'is_done': done}
        self.memory.append(transition)

    def get_random_mini_batch(self):
        """
        Gets a random sample of transitions from the replay memory.

        """
        rand_idxs = random.sample(range(len(self.memory)), self.mini_batch_size)
        mini_batch = []
        for idx in rand_idxs:
            mini_batch.append(self.memory[idx])

        return mini_batch

    def train_step(self):
        """
        Updates the model based on the mini batch

        """
        if len(self.memory) > self.mini_batch_size:
            mini_batch = self.get_random_mini_batch()

            Xs = []
            ys = []
            actions = []

            for sample in mini_batch:
                y_j = sample['reward']

                # for nonterminals, add gamma*max_a(Q(phi_{j+1})) term to y_j
                if not sample['is_done']:
                    new_observation = sample['new_observation']
                    new_obs = np.array([new_observation])
                    q_new_values = self.model.predict_target(new_obs)
                    action = np.max(q_new_values)
                    y_j += self.gamma*action

                action = np.zeros(self.num_actions)
                action[sample['action']] = 1

                observation = sample['observation']

                Xs.append(observation.copy())
                ys.append(y_j)
                actions.append(action.copy())

            Xs = np.array(Xs)
            ys = np.array(ys)
            actions = np.array(actions)

            self.model.train_step(Xs, ys, actions)

    def update_target(self):
        """
        Updates the target network with weights from the trained network.

        """
        self.model.target_update_weights()
