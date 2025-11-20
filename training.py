'''
script for training the agent for snake using various methods
'''
import os

import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy

# We now use PyTorch instead of TensorFlow
import torch

from agent import DeepQLearningAgent  # we only need the DQN agent
import json

# fix random seeds for (somewhat) reproducible runs
np.random.seed(42)
torch.manual_seed(42)

# some global variables
version = 'v17.1'

# get training configurations
with open(f'model_config/{version}.json', 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']  # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

# define no of episodes, logging frequency
episodes = 2 * (10 ** 5)
log_frequency = 500
games_eval = 8

# setup the agent (Deep Q-Learning)
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions,
                           buffer_size=buffer_size, version=version)

agent_type = 'DeepQLearningAgent'
print(f'Agent is {agent_type}')

# setup the epsilon range and decay rate for epsilon
# define reward type and update frequency, see utils for more details
epsilon, epsilon_end = 1.0, 0.01
reward_type = 'current'
sample_actions = False
n_games_training = 8 * 16
decay = 0.97

if supervised:
    # lower the epsilon since some starting policy has already been trained
    epsilon = 0.01
    # load the existing model from a supervised method or some other pretrained model
    try:
        agent.load_model(file_path=f'models/{version}', iteration=0)
    except FileNotFoundError:
        # if there is no pretrained model we just start from scratch
        pass

# use only for DeepQLearningAgent
# play some games initially to fill the buffer
if supervised:
    try:
        agent.load_buffer(file_path=f'models/{version}', iteration=1)
    except FileNotFoundError:
        pass
else:
    # setup the environment
    games = 512
    env_init = SnakeNumpy(board_size=board_size, frames=frames,
                          max_time_limit=max_time_limit, games=games,
                          frame_mode=True, obstacles=obstacles, version=version)
    ct = time.time()
    _ = play_game2(env_init, agent, n_actions, n_games=games, record=True,
                   epsilon=epsilon, verbose=True, reset_seed=False,
                   frame_mode=True, total_frames=games * 64)
    print('Playing {:d} frames took {:.2f}s'.format(games * 64, time.time() - ct))

# environments for training and evaluation
env = SnakeNumpy(board_size=board_size, frames=frames,
                 max_time_limit=max_time_limit, games=n_games_training,
                 frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames,
                  max_time_limit=max_time_limit, games=games_eval,
                  frame_mode=True, obstacles=obstacles, version=version)

# training loop
model_logs = {'iteration': [], 'reward_mean': [],
              'length_mean': [], 'games': [], 'loss': []}

for index in tqdm(range(episodes)):
    # play a batch of games and store transitions in replay buffer
    _, _, _ = play_game2(env, agent, n_actions, epsilon=epsilon,
                         n_games=n_games_training, record=True,
                         sample_actions=sample_actions, reward_type=reward_type,
                         frame_mode=True, total_frames=n_games_training,
                         stateful=True)

    # train the agent once on a mini-batch from the buffer
    loss = agent.train_agent(batch_size=64,
                             num_games=n_games_training, reward_clip=True)

    # check performance every once in a while
    if (index + 1) % log_frequency == 0:
        current_rewards, current_lengths, current_games = \
            play_game2(env2, agent, n_actions, n_games=games_eval, epsilon=-1,
                       record=False, sample_actions=False, frame_mode=True,
                       total_frames=-1, total_games=games_eval)

        model_logs['iteration'].append(index + 1)
        model_logs['reward_mean'].append(round(int(current_rewards) / current_games, 2))
        model_logs['length_mean'].append(round(int(current_lengths) / current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']] \
            .to_csv(f'model_logs/{version}.csv', index=False)

        # copy weights to target network and save models
        agent.update_target_net()
        agent.save_model(file_path=f'models/{version}', iteration=(index + 1))

        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
