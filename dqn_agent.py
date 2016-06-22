import gym
from params import parse_args
from dqn import DQN
import numpy as np
from collections import deque


def run_dqn():
    # get command line arguments, defaults set in utils.py
    agent_params, dqn_params, cnn_params, prog_params = parse_args()

    env = gym.make(agent_params['environment'])
    episodes = agent_params['episodes']
    steps = agent_params['steps']
    steps_to_update = agent_params['steps_to_update']
    skipping = agent_params['skipping']
    num_actions = env.action_space.n
    observation_shape = env.observation_space.shape
    display = prog_params['display']
    monitor = prog_params['monitor']
    verbose = prog_params['verbose']

    if verbose > 0:
        print("num actions: ", num_actions)
        print("observation_shape: ", observation_shape)

    # initialize dqn learning
    dqn = DQN(num_actions, observation_shape, dqn_params, cnn_params)

    if monitor:
        env.monitor.start('./outputs/experiment-' + agent_params['run_id'])
    last_100 = deque(maxlen=100)

    total_steps = 0
    for i_episode in range(episodes):
            observation = env.reset()
            reward_sum = 0

            for t in range(steps):
                    if display:
                        env.render()

                    # Use the previous action if in a skipping frame
                    if total_steps % skipping == 0:
                        # select action based on the model
                        action = dqn.select_action(observation)

                    # execute actin in emulator
                    new_observation, reward, done, _ = env.step(action)

                    # Only update the network if not in a skipping frame
                    if total_steps % skipping == 0:
                        # update the state
                        dqn.update_state(action, new_observation, reward, done)

                        # train the model
                        dqn.train_step()

                    observation = new_observation

                    reward_sum += reward

                    if done:
                            if verbose > 0:
                                print("Episode ", i_episode)
                            if verbose > 1:
                                print("Finished after {} timesteps".format(t+1))
                                print("Reward for this episode: ", reward_sum)
                            if verbose > 0:
                                last_100.append(reward_sum)
                                print("Average reward for last 100 episodes: ", np.mean(last_100))
                            break

                    if total_steps % steps_to_update == 0:
                        if verbose > 0:
                            print("updating target network...")
                        dqn.update_target()

                    total_steps += 1
    if monitor:
        env.monitor.close()

if __name__ == '__main__':
    run_dqn()
