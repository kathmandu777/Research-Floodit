
import matplotlib.pyplot as plt
import sys
import datetime
import gym
import numpy as np
import os
import re
import shutil
from stat import SF_IMMUTABLE

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import random

sys.path.append("../gym-floodit")
import gym_floodit  # nopep
sys.path.pop()

#!読み込みフォルダ
demo_dir_name = "002_CNN-1_E_L"  # ?実行したい学習結果のフォルダ
level = demo_dir_name[-3]
demo_steps = 2180000  # ?実行したい学習結果のステップ数 final=0

result_folder_path = "C:/Users/kator/OneDrive/ドキュメント/ResearchFloodit/data/Torch"
result_folder_path += "/Dueling"  # ?適宜変更


#!Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#!Dueling Networkを用いたQ関数を実現するためのニューラルネットワークをクラスとして記述


class CNNQNetwork(nn.Module):
    def __init__(self, state_shape, n_action):
        super(CNNQNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_action = n_action

        # ? ネットワークの構成

        # Dueling Networkでも, 畳込み部分は共有する
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=3,
                      stride=1),
            nn.ReLU(),
        )

        cnn_out_size = self.check_cnn_size(state_shape)
        print("cnn_out_size=", cnn_out_size)
        # Dueling Networkのための分岐した全結合層
        # 状態価値
        self.fc_state = nn.Sequential(
            nn.Linear(cnn_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # アドバンテージ
        self.fc_advantage = nn.Sequential(
            nn.Linear(cnn_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_action)
        )

    def check_cnn_size(self, shape):
        shape = torch.FloatTensor(1, 1, shape[1], shape[2])
        out = self.conv_layers(shape).size()
        out = out[0] * out[1] * out[2] * out[3]
        return out

    def forward(self, obs):
        feature = self.conv_layers(obs)
        feature = feature.view(feature.size(0), -1)  # Flatten. 64*11*11->7744

        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        # 状態価値 + アドバンテージ で行動価値を計算しますが、安定化のためアドバンテージの（行動間での）平均を引きます
        action_values = state_values + advantage - \
            torch.mean(advantage, dim=1, keepdim=True)
        return action_values

    # epsilon-greedy. 確率epsilonでランダムに行動し, それ以外はニューラルネットワークの予測結果に基づいてgreedyに行動します.
    def act(self, obs, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            # 行動を選択する時には勾配を追跡する必要がない
            with torch.no_grad():
                action = torch.argmax(self.forward(obs.unsqueeze(0))).item()
        return action


class TorchFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        height, width = self.observation_space.shape
        channels = 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=5,
            shape=(channels, height, width),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return torch.as_tensor(obs, dtype=torch.float).unsqueeze(0)


#!envの初期化（インスタンス作成）
env = gym.make("floodit-v0", level=demo_dir_name[-3], ignore_same_action=True)
env = TorchFrame(env)  # ラッパーの追加

#!重みの読み込み
load_path = os.path.join(result_folder_path, demo_dir_name)
if (demo_steps == 0):
    load_path += "/weights_final.pth"
else:
    load_path += "/checkpoints/weights_" + str(demo_steps) + "steps.pth"

net = CNNQNetwork(env.observation_space.shape,
                  n_action=env.action_space.n).to(device)
net.load_state_dict(torch.load(load_path))

#!demo
SEED = 42
env = gym.make(
    "floodit-v0",
    level=level,
    ignore_same_action=True,
    visualize=True,
    win_reward=10,
    lose_penalty=5,
    same_action_penalty=0.1,
    time_penalty=None,  # None=1 / (BOARDSIZE[self.level] * 2) / 10
    action_reward=None  # None=self.changed_square /(BOARDSIZE[self.level] * 2)
)
env = TorchFrame(env)  # ラッパーの追加
env.seed(SEED)
env.action_space.seed(SEED)


experiment_times = 5  # ?適宜変更

win_num = 0
lose_num = 0
log = []
for i in range(experiment_times):
    print(i, end="times\n")
    done = False
    obs = env.reset()
    log_child = []
    while not done:
        env.render()
        action = net.act(obs.to(device), epsilon=0.0)
        obs, reward, done, info = env.step(action)

        print("{:3} {:10.7f}".format(info["changed_square"], reward))
        log_child.append([reward, info])

    if (info["isWon"]):
        win_num += 1
    elif (info["isLose"]):
        lose_num += 1
    log.append(log_child)
    print()


print()
print("Result ({}times)".format(experiment_times))
print("-------------------")
win_rate = win_num / (experiment_times) * 100
lose_rate = lose_num / (experiment_times) * 100
print("win_rate:", win_rate)
print("lose_rate", lose_rate)
print("sum:", win_rate + lose_rate)


with open(os.path.join(result_folder_path, demo_dir_name) + "/result.txt", "w", encoding='UTF-8') as f:
    f.write("Result ({}times)".format(experiment_times) + "\n")
    f.write("win_rate: " + str(win_rate) + "\n")
    f.write("lose_rate: " + str(lose_rate) + "\n")
    f.write("sum: " + str(win_rate + lose_rate) + "\n")
    f.write("---------------------------------\n")
    f.write("life : changed_square, reward = result \n\n")

    for log_child in log:
        for i in range(len(log_child)):
            f.write(str(log_child[i][1]["life"]) + " : ")
            f.write("{:2}, ".format(log_child[i][1]["changed_square"]))
            f.write("{:7.4f} ".format(log_child[i][0]))  # reward
            if (log_child[i][1]["isWon"] == True):
                f.write("Win\n")
            elif (log_child[i][1]["isLose"] == True):
                f.write("Lose\n")
            else:
                f.write("\n")
        f.write("\n")
