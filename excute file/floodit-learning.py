# DQNを用いてFlood-Itを攻略

import matplotlib.pyplot as plt
import rl.callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from rl.agents.dqn import *
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import sys
import datetime
import gym
import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model
import os
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import re
import shutil
from stat import SF_IMMUTABLE

sys.path.append("../gym-floodit")
import gym_floodit  # nopep
sys.path.pop()


#!FloodItの初期化
# HIGH:H or MEDIUM:M or EASY:E
level = "M"  # ?適宜変更

env = gym.make("floodit-v0", level=level)  # envの初期化（インスタンス作成）
print("\n\n")
print("*" * 50 + "  FloodIt  " + "*" * 50)
print("action_space      : " + str(env.action_space))
print("observation_space : " + str(env.observation_space))
print("reward_range      : " + str(env.reward_range))
print()


#!各データ保存用フォルダの作成
summary = "CNN-7"  # ?毎回変更

result_folder_path = "C:/Users/kator/OneDrive/ドキュメント/ResearchFloodit/result"
result_folder_path += "/DDQN"  # ?適宜変更

file_and_folder = os.listdir(result_folder_path)
dir_list = [f for f in file_and_folder if os.path.isdir(
    os.path.join(result_folder_path, f))]
if (len(dir_list) == 0):
    last_num = 0
else:
    last_num = int(re.findall('^[0-9]+', dir_list[-1])[0])  # 先頭の数字のみ抽出
next_num = str(last_num + 1).zfill(3)
folder_name = next_num + "_" + summary + "_" + level + "_L"
folder_path = os.path.join(result_folder_path, folder_name)
os.makedirs(folder_path)
os.makedirs(os.path.join(folder_path, "checkpoints"))

shutil.copyfile(os.path.abspath(__file__), folder_path +
                "/for_record.py")  # 実行したファイルのコピーを作成
os.chmod(folder_path + "/for_record.py",
         SF_IMMUTABLE)  # change to read-only


#!モデルの定義
model = Sequential()
model.add(Reshape(env.observation_space.shape + (1,),
                  input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

print("\n\n")
print("*" * 50 + "  Model  " + "*" * 50)
print()
print(model.summary())
plot_model(model, to_file=folder_path + "/model.png", show_shapes=True)


# !Agentの定義
memory = SequentialMemory(limit=50000, window_length=1,
                          ignore_episode_boundaries=True)
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy, enable_double_dqn=True)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# !ファインチューニング[fine tuning](事前に学習した重みを使う)
# print("*" * 50 + "  fine tuning  " + "*" * 50)
# fine_tuning_weight_path=os.path.join(result_folder_path,"your_foldername/weights_final.h5f")
# print(fine_tuning_weight_path)
# print()

# dqn.load_weights(fine_tuning_weight_path)


# !学習
print("\n\n")
print("*" * 50 + "  History  " + "*" * 50)
print()

MAX_STEPS = 1000000  # 学習ステップ回数

checkpoint_weights_filename = folder_path + \
    "/checkpoints/weights_{step}steps.h5f"
logdir = folder_path


callbacks = [ModelIntervalCheckpoint(
    checkpoint_weights_filename, interval=100000)]  # ?モデルを保存する頻度

callbacks += [tf.keras.callbacks.TensorBoard(log_dir=logdir)]
history = dqn.fit(env, callbacks=callbacks, nb_steps=MAX_STEPS, visualize=False,
                  verbose=1, log_interval=100)

# 重みの保存
dqn.save_weights(folder_path + "/weights_final.h5f", overwrite=False)


# !グラフを表示
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
plt.savefig(folder_path + "/step-reward_plt.jpg")
plt.show()


# !学習結果のテスト
print("\n\n")
print("*" * 50 + "  Test  " + "*" * 50)
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
