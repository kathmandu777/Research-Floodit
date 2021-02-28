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
import sys
import datetime
import gym
import gym_floodit
import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model
import os
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

"""
パラメータ:

ネットワーク(モデル)定義
DQN agentの定義
学習ステップ回数
モデルを保存する頻度
logをとる頻度
学習回数に応じてグラフの描画の分割度(2箇所)

"""
env = gym.make("floodit-v0")  # gameの初期化（インスタンス作成）
nb_actions = env.action_space.n
now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("../result/DQN/"+now_time)
os.makedirs("../result/DQN/"+now_time+"/checkpoints")

# DQNのネットワーク定義
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# 一次元化したとすると6*6=36
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
model.add(Dense(nb_actions))
model.add(Activation('linear'))

print()
print("*"*50+"  Model  "+"*"*50)
print()
print(model.summary())
plot_model(model, to_file="../result/DQN/"+str(now_time) +
           "/DQN_model.png", show_shapes=True)


# DQN agentの定義
memory = SequentialMemory(limit=50000, window_length=1,
                          ignore_episode_boundaries=True)
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# 学習
print()
print("*" * 50 + "  History  " + "*" * 50)

MAX_STEPS = 1000000  # 学習ステップ回数

checkpoint_weights_filename = "../result/DQN/" + \
    str(now_time) + "/checkpoints/weights_{step}steps.h5f"
#log_filename = "../result/DQN/" + str(now_time) + '/log.json'
logdir = "..\\result\\DQN\\" + str(now_time)

# モデルを保存する頻度
callbacks = [ModelIntervalCheckpoint(
    checkpoint_weights_filename, interval=100000)]
# logをとる頻度(なし?)
callbacks += [tf.keras.callbacks.TensorBoard(log_dir=logdir)]

history = dqn.fit(env, callbacks=callbacks, nb_steps=MAX_STEPS, visualize=False,
                  verbose=1)


# 重みの保存
dqn.save_weights("../result/DQN/"+str(now_time) +
                 "/weights_final.h5f", overwrite=False)


# グラフを表示
# 1エピソードのstep試行回数
plt.subplot(2, 1, 1)
x = []
y = []
for i in range(len(history.history["nb_episode_steps"])):
    if(i % 1000 == 0):
        x.append(i)
        y.append(history.history["nb_episode_steps"][i])
plt.plot(x, y)
plt.ylabel("step")

# 1エピソードの報酬
plt.subplot(2, 1, 2)
x = []
y = []
for i in range(len(history.history["episode_reward"])):
    if(i % 1000 == 0):
        x.append(i)
        y.append(history.history["episode_reward"][i])
plt.plot(x, y)
plt.ylabel("reward")

plt.xlabel("episode")
plt.savefig("../result/DQN/" + now_time + str("/step-reward_plt.jpg"))
plt.show()


# 学習結果のテスト
print()
print("*"*50+"  Test  "+"*"*50)
print()
dqn.test(env, nb_episodes=10, visualize=True)


"""
参考(DQN実装方法):

keras-rl(Document)=https://keras-rl.readthedocs.io/en/latest/agents/overview/
    https://qiita.com/harmegiddo/items/4226df13139d6ba34018
    https://qiita.com/ohtaman/items/edcb3b0a2ff9d48a7def
    https://qiita.com/inoory/items/e63ade6f21766c7c2393
    https://qiita.com/pocokhc/items/a8120b0abd5941dd7a9f#keras-rl-%E3%81%AE%E5%AD%A6%E7%BF%92%E9%81%8E%E7%A8%8B%E3%81%AE%E5%8F%AF%E8%A6%96%E5%8C%96%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB
    https://kagglenote.com/ml-tips/my-environment-with-gym/

詳細:
    https://qiita.com/goodclues/items/9b2b618ac5ba4c3be1c5

misc:
    https://www.tcom242242.net/entry/python-basic/keras-rl/%E3%80%90keras-rlcolab%E3%80%91keras-rl%E3%82%92colab%E3%81%AEgpu%E3%82%92%E7%94%A8%E3%81%84%E3%81%A6%E5%AE%9F%E8%A1%8C%E3%81%99%E3%82%8B/
    https://github.com/ibab/tensorflow-wavenet/issues/255
"""
