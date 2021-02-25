import keras
from keras.models import Sequential, Model
from keras.layers import *

from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import numpy as np
import FloodIt_gym

# gym
env = FloodIt_gym.FloodItEnv()
print("-Initial parameter-")
print(env.action_space)  # input
print(env.observation_space)  # output
print(env.reward_range)  # rewards
print(env.action_space)  # action
print(env.action_space.sample())  # action
print("-------------------")


# model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# experience replay用のmemory
memory = SequentialMemory(limit=50000, window_length=1)
# 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

history = dqn.fit(env, nb_steps=50000, visualize=False,
                  verbose=2, nb_max_episode_steps=300)
# 学習の様子を描画したいときは、Envに_render()を実装して、visualize=True にします,
