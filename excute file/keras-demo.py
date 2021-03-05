
import matplotlib.pyplot as plt
import rl.callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import sys
import datetime
import gym
import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model
import os
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

sys.path.append("../gym-floodit")
import gym_floodit  # nopep
sys.path.pop()


demo_dir_name = "002_NN-6_E_L"  # ?実行したい学習結果のフォルダ
demo_steps = 0  # ?実行したい学習結果のステップ数 final=0

result_folder_path = "C:/Users/kator/OneDrive/ドキュメント/ResearchFloodit/data/keras"
result_folder_path += "/DQN"  # ?適宜変更

# ?同じ手の時にリセットすかどうか
env = gym.make("floodit-v0", level=demo_dir_name[-3], ignore_same_action=True)

# ?学習時のモデルの定義をコピペ
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))


# ?学習時のAgentの定義をコピペ
memory = SequentialMemory(limit=50000, window_length=1,
                          ignore_episode_boundaries=True)
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

load_path = os.path.join(result_folder_path, demo_dir_name)
if (demo_steps == 0):
    load_path += "/weights_final.h5f"
else:
    load_path += "/checkpoints/weights_" + str(demo_steps) + "steps.h5f"
dqn.load_weights(load_path)


dqn.test(env, nb_episodes=20, visualize=True)  # ?何回実行したいか
