
import matplotlib.pyplot as plt
import rl.callbacks
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import sys
import datetime
import gym
import gym_floodit
import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model
import os
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


env = gym.make("floodit-v0")  # gameの初期化（インスタンス作成）
nb_actions = env.action_space.n


# DQNのネットワーク定義
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))

# 一次元化したとすると6*6=36
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))


# DQN agentの定義
memory = SequentialMemory(limit=50000, window_length=1,
                          ignore_episode_boundaries=True)
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights("../result/DQN/20210224_163809/weights_final.h5f")


# 学習結果のテスト

dqn.test(env, nb_episodes=20, visualize=True)
