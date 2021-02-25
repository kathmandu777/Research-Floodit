# DQNを用いてFloodit1を攻略

import matplotlib.pyplot as plt
import rl.callbacks
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.callbacks import CheckpointCallback
import gym
import gym_floodit
import sys


sys.path.append('../')
env = gym.make("floodit-v0")  # gameの初期化（インスタンス作成）

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="log")
print('start learning')
checkpoint_callback = CheckpointCallback(
    save_freq=500, save_path='DQN-easy/', name_prefix='rl_model')
model.learn(total_timesteps=100000, callback=checkpoint_callback)
print('finish learning')
