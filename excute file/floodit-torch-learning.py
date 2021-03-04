# DQNを用いてFlood-Itを攻略

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


#!FloodItの初期化
# HIGH:H or MEDIUM:M or EASY:E
level = "H"  # ?適宜変更


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


env = gym.make("floodit-v0", level=level)  # envの初期化（インスタンス作成）
env = TorchFrame(env)

print("\n\n")
print("*" * 50 + "  FloodIt  " + "*" * 50)
print("action_space      : " + str(env.action_space))
print("observation_space : " + str(env.observation_space))
print("reward_range      : " + str(env.reward_range))
print()


#!各データ保存用フォルダの作成
summary = "CNN-7"  # ?毎回変更

result_folder_path = "C:/Users/kator/OneDrive/ドキュメント/ResearchFloodit/result"
result_folder_path += "/Torch"  # ?適宜変更

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


#!Torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
   Prioritized Experience Replayを実現するためのメモリクラス.
"""


class PrioritizedReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.priorities[0] = 1.0

    def __len__(self):
        return len(self.buffer)

    # 経験をリプレイバッファに保存する． 経験は(obs, action, reward, next_obs, done)の5つ組を想定
    def push(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience

        # 優先度は最初は大きな値で初期化しておき, 後でサンプルされた時に更新する
        self.priorities[self.index] = self.priorities.max()
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        # 現在経験が入っている部分に対応する優先度を取り出し, サンプルする確率を計算
        priorities = self.priorities[: self.buffer_size if len(
            self.buffer) == self.buffer_size else self.index]
        priorities = priorities ** alpha
        prob = priorities / priorities.sum()

        # 上で計算した確率に従ってリプレイバッファ中のインデックスをサンプルする
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)

        # 学習の方向性を補正するための重みを計算
        weights = (len(self.buffer) * prob[indices]) ** (-beta)
        weights = weights / weights.max()

        # 上でサンプルしたインデックスに基づいて経験をサンプルし, (obs, action, reward, next_obs, done)に分ける
        obs, action, reward, next_obs, done = zip(
            *[self.buffer[i] for i in indices])

        # あとで計算しやすいようにtorch.Tensorに変換して(obs, action, reward, next_obs, done, indices, weights)の7つ組を返す
        return (torch.stack(obs),
                torch.as_tensor(action),
                torch.as_tensor(reward, dtype=torch.float32),
                torch.stack(next_obs),
                torch.as_tensor(done, dtype=torch.uint8),
                indices,
                torch.as_tensor(weights, dtype=torch.float32))

    # 優先度を更新する. 優先度が極端に小さくなって経験が全く選ばれないということがないように, 微小値を加算しておく.
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + 1e-4


"""
    Dueling Networkを用いたQ関数を実現するためのニューラルネットワークをクラスとして記述します. 
"""


class CNNQNetwork(nn.Module):
    def __init__(self, state_shape, n_action):
        super(CNNQNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_action = n_action
        # Dueling Networkでも, 畳込み部分は共有する
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=4,
                      stride=2),  # 1x30x30 -> 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # 32x14x14 -> 64*12*12
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),  # 64*12*12 -> 64*11*11
            nn.ReLU()
        )

        # Dueling Networkのための分岐した全結合層
        # 状態価値
        self.fc_state = nn.Sequential(
            nn.Linear(7744, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # アドバンテージ
        self.fc_advantage = nn.Sequential(
            nn.Linear(7744, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )

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


"""
    リプレイバッファの宣言
"""
buffer_size = 100000  # 　リプレイバッファに入る経験の最大数
initial_buffer_size = 10000  # 学習を開始する最低限の経験の数
replay_buffer = PrioritizedReplayBuffer(buffer_size)


"""
    ネットワークの宣言
"""
net = CNNQNetwork(env.observation_space.shape,
                  n_action=env.action_space.n).to(device)
target_net = CNNQNetwork(env.observation_space.shape,
                         n_action=env.action_space.n).to(device)
target_update_interval = 2000  # 学習安定化のために用いるターゲットネットワークの同期間隔


"""
    オプティマイザとロス関数の宣言
"""
optimizer = optim.Adam(net.parameters(), lr=1e-4)  # オプティマイザはAdam
loss_func = nn.SmoothL1Loss(reduction='none')  # ロスはSmoothL1loss（別名Huber loss）


"""
    Prioritized Experience Replayのためのパラメータβ
"""
beta_begin = 0.4
beta_end = 1.0
beta_decay = 500000
# beta_beginから始めてbeta_endまでbeta_decayかけて線形に増やす


def beta_func(step): return min(beta_end, beta_begin +
                                (beta_end - beta_begin) * (step / beta_decay))


"""
    探索のためのパラメータε
"""
epsilon_begin = 1.0
epsilon_end = 0.01
epsilon_decay = 50000
# epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす


def epsilon_func(step): return max(epsilon_end, epsilon_begin -
                                   (epsilon_begin - epsilon_end) * (step / epsilon_decay))


"""
    その他のハイパーパラメータ
"""
gamma = 0.99  # 割引率
batch_size = 32
n_episodes = 300  # 学習を行うエピソード数


def update(batch_size, beta):
    obs, action, reward, next_obs, done, indices, weights = replay_buffer.sample(
        batch_size, beta)
    obs, action, reward, next_obs, done, weights \
        = obs.to(device), action.to(device), reward.to(device), next_obs.to(device), done.to(device), weights.to(device)

    # ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
    q_values = net(obs).gather(1, action.unsqueeze(1)).squeeze(1)

    # 目標値の計算なので勾配を追跡しない
    with torch.no_grad():
        # Double DQN.
        # ① 現在のQ関数でgreedyに行動を選択し,
        greedy_action_next = torch.argmax(net(next_obs), dim=1)
        # ② 対応する価値はターゲットネットワークのものを参照します.
        q_values_next = target_net(next_obs).gather(
            1, greedy_action_next.unsqueeze(1)).squeeze(1)

    # ベルマン方程式に基づき, 更新先の価値を計算します.
    # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
    target_q_values = reward + gamma * q_values_next * (1 - done)

    # Prioritized Experience Replayのために, ロスに重み付けを行なって更新します.
    optimizer.zero_grad()
    loss = (weights * loss_func(q_values, target_q_values)).mean()
    loss.backward()
    optimizer.step()

    # TD誤差に基づいて, サンプルされた経験の優先度を更新します.
    replay_buffer.update_priorities(
        indices, (target_q_values - q_values).abs().detach().cpu().numpy())

    return loss.item()


# TensorBoardをColab内に起動. リアルタイムに学習経過が更新されます
writer = SummaryWriter('./logs')


#!学習
step = 0
for episode in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedyで行動を選択
        action = net.act(obs.to(device), epsilon_func(step))
        # 環境中で実際に行動
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward

        # リプレイバッファに経験を蓄積
        replay_buffer.push([obs, action, reward, next_obs, done])
        obs = next_obs

        # ネットワークを更新
        if len(replay_buffer) > initial_buffer_size:
            update(batch_size, beta_func(step))

        # ターゲットネットワークを定期的に同期させる
        if (step + 1) % target_update_interval == 0:
            target_net.load_state_dict(net.state_dict())

        step += 1

    print('Episode: {},  Step: {},  Reward: {}'.format(
        episode + 1, step + 1, total_reward))
    writer.add_scalar('Reward', total_reward, episode)

writer.close()


#!学習結果の確認
# envの初期化（インスタンス作成）
env = gym.make("floodit-v0", level=level, ignore_same_action=True)
env = TorchFrame(env)

for _ in range(100):
    env.render()
    action = net.act(obs.to(device), epsilon=0.0)
    obs, reward, done, info = env.step(action)

    if done:
        env.render()
        obs = env.reset()
